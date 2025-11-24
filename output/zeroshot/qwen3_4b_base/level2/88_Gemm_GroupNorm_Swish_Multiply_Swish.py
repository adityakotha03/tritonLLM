import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    x_ptr, 
    w_ptr, 
    out_ptr, 
    batch_size: tl.constexpr, 
    in_features: tl.constexpr, 
    out_features: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute the block of output to process
    pid = tl.program_id(0)
    block_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n = tl.arange(0, BLOCK_SIZE_N)

    # Load weights in a column-major fashion (for efficient GEMM)
    # We assume w is (out_features, in_features)
    w = tl.load(w_ptr + block_n[:, None] * in_features + block_m[None, :], mask=block_m[:, None] < out_features, other=0.0)

    # Compute the output for each row of x
    x = tl.load(x_ptr + block_m[:, None] * in_features + tl.arange(0, in_features), mask=block_m[:, None] < batch_size, other=0.0)
    # Perform GEMM: out[i] = sum_j x[j] * w[i, j]
    out = tl.dot(x, w)
    tl.store(out_ptr + block_m[:, None] * out_features + block_n[None, :], out, mask=block_m[:, None] < batch_size)


@triton.jit
def group_norm_kernel(
    x_ptr, 
    g_norm_ptr, 
    out_ptr, 
    batch_size: tl.constexpr, 
    out_features: tl.constexpr, 
    num_groups: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a contiguous block of output features
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_features

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute group-wise statistics (mean and variance)
    # We assume input is (batch_size, out_features)
    # We compute mean and variance per group
    group_size = out_features // num_groups
    group_id = offsets // group_size
    group_offset = offsets % group_size

    # Reduce over the group dimension
    group_mean = tl.zeros((group_size,), dtype=tl.float32)
    group_var = tl.zeros((group_size,), dtype=tl.float32)

    # Compute mean and variance for each group
    for i in range(group_size):
        # Load the element in this group
        idx = group_offset + i
        if idx < out_features:
            val = x[idx]
            group_mean += val
            group_var += val * val
    
    # Normalize
    group_mean = group_mean / group_size
    group_var = group_var / group_size
    group_std = tl.sqrt(group_var + 1e-5)

    # Apply normalization
    out = (x - group_mean) / group_std
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def swish_kernel(
    x_ptr, 
    out_ptr, 
    n_elements: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def multiply_kernel(
    x_ptr, 
    w_ptr, 
    out_ptr, 
    n_elements: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    out = x * w
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gemm(x: torch.Tensor, w: torch.Tensor):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Use FP16 for GEMM to leverage Tensor Core
    x_fp16 = x.half()
    w_fp16 = w.half()
    out = torch.empty((batch_size, out_features), dtype=torch.float16, device=x.device)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Grid size
    grid = lambda meta: ((out_features + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],)

    # Launch kernel
    gemm_kernel[grid](x_fp16, w_fp16, out, batch_size, in_features, out_features, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return out


def triton_group_norm(x: torch.Tensor, num_groups: int, out_features: int):
    assert x.is_cuda and x.device.type == 'cuda', "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = lambda meta: ((out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    group_norm_kernel[grid](x, None, out, x.shape[0], out_features, num_groups, BLOCK_SIZE)
    return out


def triton_swish(x: torch.Tensor):
    assert x.is_cuda and x.device.type == 'cuda', "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    swish_kernel[grid](x, out, x.numel(), BLOCK_SIZE)
    return out


def triton_multiply(x: torch.Tensor, w: torch.Tensor):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    multiply_kernel[grid](x, w, out, x.numel(), BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        # We keep the parameter as a nn.Parameter
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape, dtype=torch.float16))

        # We do not use nn.Linear anymore; instead, we use custom kernels
        # The GEMM is now implemented via Triton

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = triton_gemm(x, self.multiply_weight.t())  # Note: we transpose weight for GEMM

        # (batch_size, out_features) -> (batch_size, out_features)
        x = triton_group_norm(x, num_groups, x.shape[1])

        # (batch_size, out_features) -> (batch_size, out_features)
        x = triton_swish(x)

        # (batch_size, out_features) -> (batch_size, out_features)
        x = triton_multiply(x, self.multiply_weight)

        # (batch_size, out_features) -> (batch_size, out_features)
        x = triton_swish(x)

        return x