import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_add_swish_kernel(
    a_ptr, b_ptr, add_ptr, out_ptr,
    n_rows, n_cols, n_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    USE_ADD: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Matrix multiplication: A @ B + add_value
    # A: [n_rows, n_features], B: [n_features, n_cols], add_value: [n_cols]

    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offset for block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Offsets for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Mask to avoid out-of-bounds access
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask_k = offs_k < n_features

    # Accumulator for dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform matrix multiplication with tiling
    for k in range(0, n_features, BLOCK_SIZE_K):
        # Load A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        offs_k_current = k + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * n_features + offs_k_current[None, :],
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        )

        # Load B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b = tl.load(
            b_ptr + offs_k_current[:, None] * n_cols + offs_n[None, :],
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0
        )

        # Accumulate result
        accumulator += tl.dot(a, b)

    # Convert to float32 for activation
    accumulator = accumulator.to(tl.float32)

    # Add bias if enabled
    if USE_ADD:
        add_val = tl.load(
            add_ptr + offs_n,
            mask=mask_n,
            other=0.0
        )
        accumulator += add_val[None, :]

    # Apply Swish: x * sigmoid(x)
    if ACTIVATION == 1:
        # sigmoid(x) = 1 / (1 + exp(-x))
        exp_x = tl.exp(-accumulator)
        sigmoid_x = 1.0 / (1.0 + exp_x)
        output = accumulator * sigmoid_x

    # Store result
    tl.store(
        out_ptr + offs_m[:, None] * n_cols + offs_n[None, :],
        output,
        mask=(mask_m[:, None] & mask_n[None, :])
    )


@triton.jit
def tanh_gelu_hardtanh_kernel(
    x_ptr, out_ptr,
    n_elements, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Process each block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply Tanh
    tanh_x = tl.tanh(x)

    # Apply GELU approximation: x * 0.5 * (1 + tanh(sqrt(pi/2) * (x + 0.044715 * x^3)))
    pi_over_2 = 1.7724538509055159
    coeff = 0.044715
    sqrt_pi_over_2 = tl.sqrt(pi_over_2)
    x_cubed = x * x * x
    gelu_input = sqrt_pi_over_2 * (x + coeff * x_cubed)
    gelu_result = 0.5 * (1.0 + tl.tanh(gelu_input))
    gelu_x = x * gelu_result

    # Apply Hardtanh: clamp to [-1, 1]
    hardtanh_x = tl.clamp(gelu_x, -1.0, 1.0)

    # Store output
    tl.store(out_ptr + offsets, hardtanh_x, mask=mask)


def triton_matmul_add_swish(x, weight, bias):
    # Ensure contiguous tensors on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output shape
    n_rows, n_cols = x.shape[0], weight.shape[1]

    # Parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid setup for matmul
    grid = lambda meta: (
        triton.cdiv(n_rows, meta["BLOCK_SIZE_M"]),
        triton.cdiv(n_cols, meta["BLOCK_SIZE_N"])
    )

    # Launch kernel
    out = torch.empty_like(x, dtype=torch.float32)
    matmul_add_swish_kernel[grid](
        x, weight, bias, out,
        x.shape[0], weight.shape[1], x.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_ADD=True,
        ACTIVATION=1
    )
    return out


def triton_tanh_gelu_hardtanh(x):
    x = x.contiguous()
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x, dtype=torch.float32)
    tanh_gelu_hardtanh_kernel[grid](x, out, n_elements, x.shape[0], x.shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        # Fuse matmul, add, and swish
        x = triton_matmul_add_swish(x, self.weight, self.bias)

        # Fuse tanh, gelu, hardtanh
        x = triton_tanh_gelu_hardtanh(x)

        return x