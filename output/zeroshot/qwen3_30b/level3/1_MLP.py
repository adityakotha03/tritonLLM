import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offsets for the current block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create ranges for the block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Mask to avoid out-of-bounds access
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_n[None, :] & mask_k[None, :]

    # Load data from global memory
    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=mask,
        other=0.0
    )
    b = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=mask,
        other=0.0
    )

    # Perform matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b)

    # Apply activation if required
    if ACTIVATION == "ReLU":
        acc = tl.max(acc, 0.0)
    elif ACTIVATION == "GELU":
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = 0.7978845608028654
        x = acc
        x3 = x * x * x
        tanh_input = sqrt_2_pi * (x + 0.044715 * x3)
        tanh_val = tl.tanh(tanh_input)
        acc = 0.5 * x * (1.0 + tanh_val)

    # Store result
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=mask
    )


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, activation=""):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication"

    out = torch.empty(M, N, dtype=a.dtype, device=a.device)

    # Grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        triton.cdiv(K, meta["BLOCK_SIZE_K"])
    )

    # Determine block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Launch kernel
    matmul_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=activation
    )

    return out


def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    y = y.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            # Replace Linear + ReLU with fused Triton Matmul + ReLU
            layers.append(
                nn.Linear(current_input_size, layer_size).cuda()
            )
            layers.append(nn.ReLU())
            current_input_size = layer_size
        
        # Final linear layer
        layers.append(
            nn.Linear(current_input_size, output_size).cuda()
        )
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Use Triton kernels for matmul and add (bias)
        # Extract the model's parameters
        layer_idx = 0
        current_x = x

        for module in self.network:
            if isinstance(module, nn.Linear):
                # Use Triton for matmul
                weight = module.weight
                bias = module.bias
                # Convert to float32 for Triton kernel
                current_x = triton_matmul(current_x, weight.t(), activation="ReLU")
                # Add bias
                if bias is not None:
                    current_x = triton_add(current_x, bias)
            elif isinstance(module, nn.ReLU):
                # ReLU already fused in matmul, skip
                continue
            else:
                raise ValueError(f"Unsupported module: {module}")
        
        return current_x