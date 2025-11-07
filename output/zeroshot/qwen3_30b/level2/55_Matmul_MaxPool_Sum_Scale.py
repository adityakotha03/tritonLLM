import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    stride_a_batch, stride_a_in, stride_b_in, stride_b_out, stride_out_batch, stride_out_out,
    ACTIVATION: tl.constexpr,
    mask: tl.constexpr
):
    pid = tl.program_id(0)
    block_id = pid
    num_blocks = (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # Calculate the starting index for the current block
    block_start_m = (block_id // num_blocks) * BLOCK_SIZE_M
    block_start_n = (block_id % num_blocks) * BLOCK_SIZE_N

    # Define offsets for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Calculate the current K offset
        k_offset = k + offs_k
        k_mask = k_offset < in_features

        # Load A and B blocks
        a = tl.load(
            a_ptr + stride_a_batch * offs_m[:, None] + stride_a_in * k_offset[None, :],
            mask=tl.expand_mask(k_mask, (BLOCK_SIZE_M, BLOCK_SIZE_K)),
            other=0.0
        )
        b = tl.load(
            b_ptr + stride_b_in * k_offset[:, None] + stride_b_out * offs_n[None, :],
            mask=tl.expand_mask(k_mask, (BLOCK_SIZE_K, BLOCK_SIZE_N)),
            other=0.0
        )

        # Perform matmul
        accumulator += tl.dot(a, b)

    # Apply activation if needed (ReLU)
    if ACTIVATION:
        accumulator = tl.maximum(accumulator, 0.0)

    # Save output
    out_m = offs_m < batch_size
    out_n = offs_n < out_features
    mask = tl.expand_mask(out_m, (BLOCK_SIZE_M, 1)) & tl.expand_mask(out_n, (1, BLOCK_SIZE_N))
    tl.store(
        out_ptr + stride_out_batch * offs_m[:, None] + stride_out_out * offs_n[None, :],
        accumulator,
        mask=mask
    )


@triton.jit
def max_pool_kernel(
    x_ptr, out_ptr,
    batch_size, in_features, out_features, kernel_size,
    BLOCK_SIZE: tl.constexpr,
    stride_x_batch, stride_x_in, stride_out_batch, stride_out_out,
    mask: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the effective output features
    out_len = in_features // kernel_size
    in_offsets = offs * kernel_size + tl.arange(0, kernel_size)[:, None]
    in_mask = in_offsets < in_features

    # Load input and compute max
    x = tl.load(x_ptr + stride_x_batch * offs[:, None] + stride_x_in * in_offsets, mask=in_mask)
    out = tl.max(x, axis=0)

    # Store output
    out_mask = offs < out_len
    tl.store(out_ptr + stride_out_batch * offs + stride_out_out * tl.arange(0, out_features), out, mask=out_mask)


@triton.jit
def sum_and_scale_kernel(
    x_ptr, out_ptr,
    batch_size, out_features,
    BLOCK_SIZE: tl.constexpr,
    stride_x_batch, stride_x_out, stride_out_batch, stride_out_out,
    scale_factor: tl.constexpr,
    mask: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Check bounds
    valid = offs < batch_size

    # Sum along dim 1
    x = tl.load(x_ptr + stride_x_batch * offs[:, None] + stride_x_out * tl.arange(0, out_features)[None, :], mask=valid[:, None])
    sums = tl.sum(x, axis=1)

    # Apply scale
    out = sums * scale_factor

    # Store result
    tl.store(out_ptr + stride_out_batch * offs + stride_out_out * tl.arange(0, out_features), out, mask=valid)


def triton_matmul(x, w, activation=True):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[1]

    # Prepare output
    out = torch.empty(batch_size, out_features, device=x.device, dtype=torch.float32)

    # Set block sizes
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64

    # Calculate grid
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],)

    # Launch kernel
    matmul_kernel[grid](
        x, w, out,
        batch_size, in_features, out_features,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        activation,
        BLOCK_SIZE_M * BLOCK_SIZE_N
    )
    return out


def triton_max_pool(x, kernel_size):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    batch_size, in_features = x.shape
    out_features = in_features // kernel_size

    # Prepare output
    out = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)

    # Block size
    BLOCK_SIZE = 128

    # Grid
    grid = lambda meta: ((out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    max_pool_kernel[grid](
        x, out,
        batch_size, in_features, out_features, kernel_size,
        BLOCK_SIZE,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE
    )
    return out


def triton_sum_and_scale(x, scale_factor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    batch_size, out_features = x.shape

    # Prepare output
    out = torch.empty(batch_size, device=x.device, dtype=x.dtype)

    # Block size
    BLOCK_SIZE = 256

    # Grid
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    sum_and_scale_kernel[grid](
        x, out,
        batch_size, out_features,
        BLOCK_SIZE,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        scale_factor,
        BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.linear_weight = nn.Parameter(torch.randn(out_features, in_features, device='cuda', dtype=torch.float32))
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        # Matmul + ReLU fused
        x = triton_matmul(x, self.linear_weight, activation=True)
        
        # Expand dimension for MaxPool1d
        x = x.unsqueeze(1)
        
        # Max pool (fused with stride)
        x = triton_max_pool(x, self.kernel_size)
        
        # Squeeze back
        x = x.squeeze(1)
        
        # Sum along dim 1 and scale
        x = triton_sum_and_scale(x, self.scale_factor)
        
        return x