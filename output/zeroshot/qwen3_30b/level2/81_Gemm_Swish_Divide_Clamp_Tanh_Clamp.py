import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    stride_x, stride_w, stride_b, stride_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION_SCALE: tl.constexpr
):
    # Map program to data
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(in_features, BLOCK_SIZE_K)
    num_pid = num_pid_m * num_pid_n * num_pid_k

    # Calculate the PID in each dimension
    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid // num_pid_k) % num_pid_n
    pid_k = pid % num_pid_k

    # Calculate offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create offsets for current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k blocks
    for _ in range(num_pid_k):
        # Load x and w blocks
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x + offs_k[None, :] * stride_x,
            mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features),
            other=0.0
        )
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_w + offs_n[None, :] * stride_w,
            mask=(offs_k[:, None] < in_features) & (offs_n[None, :] < out_features),
            other=0.0
        )
        # Perform matrix multiplication
        accumulator += tl.dot(x, w)

        # Advance k offset
        block_start_k += BLOCK_SIZE_K

    # Add bias if exists
    bias = tl.load(
        b_ptr + offs_n,
        mask=offs_n < out_features,
        other=0.0
    )
    accumulator += bias[None, :]

    # Convert to bf16 for swish and later ops
    accumulator = accumulator.to(tl.bfloat16)

    # Apply swish: x * sigmoid(x)
    # Use online sigmoid for numerical stability
    # We split into: x * (1 / (1 + exp(-x))) -> optimize by computing -x first
    x_swish = accumulator
    x_neg = -x_swish
    # Stable sigmoid: 1 / (1 + exp(-x)) -> 1 / (1 + exp(-x))
    # Use online softmax approximation: use stable logsumexp
    sigmoid_val = tl.math.sigmoid(x_neg)
    x_swish = x_swish * sigmoid_val

    # Divide by 2
    x_swish = x_swish * (1.0 / 2.0)

    # Clamp to [-1, 1]
    x_swish = tl.clip(x_swish, -1.0, 1.0)

    # Apply tanh
    x_tanh = tl.tanh(x_swish)

    # Clamp again to [-1, 1]
    x_tanh = tl.clip(x_tanh, -1.0, 1.0)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out + offs_n[None, :] * stride_out)
    tl.store(out_ptrs, x_tanh, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features))


def triton_gemm_swish_clamp(x, w, b=None, ACTIVATION_SCALE=0.5):
    """
    Triton implementation of Linear + Swish + Divide + Clamp + Tanh + Clamp
    with operator fusion and bf16 tensor core usage.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    # Get shapes
    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Strides
    stride_x = x.stride(0) if x.stride(0) > 1 else x.stride(1)
    stride_w = w.stride(1) if w.stride(0) == in_features else w.stride(0)
    stride_b = b.stride(0) if b is not None else 0
    stride_out = x.stride(0) if x.stride(0) > 1 else x.stride(1)

    # Choose block sizes
    # Use 256x256x256 for A100, optimized for Tensor Cores
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256

    # Calculate grid
    grid = lambda meta: (
        triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]) *
        triton.cdiv(out_features, meta["BLOCK_SIZE_N"]) *
        triton.cdiv(in_features, meta["BLOCK_SIZE_K"]),
    )

    # Launch kernel
    out = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    gemm_swish_kernel[grid](
        x, w, b, out,
        batch_size, in_features, out_features,
        stride_x, stride_w, stride_b, stride_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION_SCALE=ACTIVATION_SCALE
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
    ],
    key=['batch_size', 'in_features', 'out_features'],
)
@triton.jit
def gemm_swish_kernel_autotuned(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    stride_x, stride_w, stride_b, stride_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION_SCALE: tl.constexpr
):
    # Map program to data
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(in_features, BLOCK_SIZE_K)
    num_pid = num_pid_m * num_pid_n * num_pid_k

    # Calculate the PID in each dimension
    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid // num_pid_k) % num_pid_n
    pid_k = pid % num_pid_k

    # Calculate offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create offsets
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k blocks
    for _ in range(num_pid_k):
        # Load x and w blocks with masking
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x + offs_k[None, :] * stride_x,
            mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features),
            other=0.0
        )
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_w + offs_n[None, :] * stride_w,
            mask=(offs_k[:, None] < in_features) & (offs_n[None, :] < out_features),
            other=0.0
        )
        # Matrix multiply with Tensor Cores (fused in bfloat16)
        accumulator += tl.dot(x, w)

        block_start_k += BLOCK_SIZE_K

    # Add bias
    if b_ptr:
        bias = tl.load(
            b_ptr + offs_n,
            mask=offs_n < out_features,
            other=0.0
        )
        accumulator += bias[None, :]

    # Convert to bfloat16
    accumulator = accumulator.to(tl.bfloat16)

    # Apply swish: x * sigmoid(x)
    # Use stable sigmoid
    x_swish = accumulator
    x_neg = -x_swish
    sigmoid_val = tl.math.sigmoid(x_neg)
    x_swish = x_swish * sigmoid_val

    # Divide by 2
    x_swish = x_swish * (1.0 / 2.0)

    # Clamp to [-1, 1]
    x_swish = tl.clip(x_swish, -1.0, 1.0)

    # Apply tanh
    x_tanh = tl.tanh(x_swish)

    # Final clamp
    x_tanh = tl.clip(x_tanh, -1.0, 1.0)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out + offs_n[None, :] * stride_out)
    tl.store(out_ptrs, x_tanh, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features))


def triton_gemm_swish_clamp_autotuned(x, w, b=None, ACTIVATION_SCALE=0.5):
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Strides
    stride_x = x.stride(0) if x.stride(0) > 1 else x.stride(1)
    stride_w = w.stride(1) if w.stride(0) == in_features else w.stride(0)
    stride_b = b.stride(0) if b is not None else 0
    stride_out = x.stride(0) if x.stride(0) > 1 else x.stride(1)

    # Choose grid
    grid = lambda meta: (
        triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]) *
        triton.cdiv(out_features, meta["BLOCK_SIZE_N"]) *
        triton.cdiv(in_features, meta["BLOCK_SIZE_K"]),
    )

    out = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    gemm_swish_kernel_autotuned[grid](
        x, w, b, out,
        batch_size, in_features, out_features,
        stride_x, stride_w, stride_b, stride_out,
        ACTIVATION_SCALE=ACTIVATION_SCALE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        # Use the Triton-optimized fused kernel
        self.use_triton = True

    def forward(self, x):
        if self.use_triton:
            # Use fused Triton kernel for Linear + Swish + Divide + Clamp + Tanh + Clamp
            return triton_gemm_swish_clamp_autotuned(x, self.gemm.weight, self.gemm.bias)
        else:
            x = self.gemm(x)
            x = x * torch.sigmoid(x)
            x = x / 2.0
            x = torch.clamp(x, min=-1.0, max=1.0)
            x = torch.tanh(x)
            x = torch.clamp(x, min=-1.0, max=1.0)
            return x