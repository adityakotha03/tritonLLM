import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_relu_div_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride, w_stride, out_stride,
    batch_size, in_features, out_features,
    divisor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Map program to block of data
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Create offsets for this block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create offsets within the block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Mask for valid indices
    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features
    mask_k = offs_k < in_features

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over k blocks
    for k in range(0, (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        # Compute current k offset
        k_offset = k * BLOCK_SIZE_K
        k_mask = k_offset + tl.arange(0, BLOCK_SIZE_K) < in_features

        # Load x and w with masking
        x = tl.load(
            x_ptr + (offs_m[:, None] * x_stride + offs_k[None, :] * 1),
            mask=(mask_m[:, None] & k_mask[None, :]),
            other=0.0
        )
        w = tl.load(
            w_ptr + (offs_k[:, None] * w_stride + offs_n[None, :] * 1),
            mask=(k_mask[:, None] & mask_n[None, :]),
            other=0.0
        )

        # Perform matrix multiplication
        acc += tl.dot(x, w)

    # Apply ReLU and division
    out = tl.maximum(acc, 0.0) / divisor

    # Store output
    tl.store(
        out_ptr + (offs_m[:, None] * out_stride + offs_n[None, :] * 1),
        out,
        mask=(mask_m[:, None] & mask_n[None, :])
    )


def triton_linear_relu_div(x: torch.Tensor, w: torch.Tensor, divisor: float):
    # Ensure inputs are on GPU and contiguous
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Get dimensions
    batch_size, in_features = x.shape
    out_features = w.shape[1]

    # Allocate output
    out = torch.empty_like(x)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid dimensions
    grid_m = triton.cdiv(batch_size, BLOCK_SIZE_M)
    grid_n = triton.cdiv(out_features, BLOCK_SIZE_N)
    grid_k = triton.cdiv(in_features, BLOCK_SIZE_K)

    grid = (grid_m, grid_n, grid_k)

    # Launch kernel
    linear_relu_div_kernel[grid](
        x, w, out,
        x.stride(0), w.stride(0), out.stride(0),
        batch_size, in_features, out_features,
        divisor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.divisor = divisor

    def forward(self, x):
        # Use Triton-optimized fused linear + ReLU + division
        return triton_linear_relu_div(x, self.linear.weight, self.divisor)