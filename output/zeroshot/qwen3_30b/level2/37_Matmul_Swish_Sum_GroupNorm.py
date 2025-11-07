import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_add_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Define program ID and block offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Calculate offsets within the block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create indices for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Mask for valid indices
    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features
    mask_k = offs_k < in_features

    # Load input and weight tiles
    x = tl.load(x_ptr + offs_m[:, None] * in_features + offs_k[None, :], 
                mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    w = tl.load(w_ptr + offs_k[:, None] * out_features + offs_n[None, :], 
                mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Perform matrix multiplication with accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        x = tl.load(x_ptr + offs_m[:, None] * in_features + offs_k[None, :], 
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * out_features + offs_n[None, :], 
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        offs_k += BLOCK_SIZE_K

    # Apply bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    # Apply Swish activation: x * sigmoid(x)
    if ACTIVATION == 1:
        # Compute sigmoid using stable approach
        max_val = tl.maximum(acc, -10.0)
        exp_neg = tl.exp(-max_val)
        sigmoid = 1.0 / (1.0 + exp_neg)
        acc = acc * sigmoid

    # Store result
    tl.store(out_ptr + offs_m[:, None] * out_features + offs_n[None, :], 
             acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def group_norm_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, out_features, num_groups, group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute group size
    group_size = out_features // num_groups

    # Program IDs
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # Group index

    # Offsets within group
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Mask for valid indices
    mask_m = offs_m < batch_size
    mask_n = offs_n < group_size

    # Calculate group start
    group_start = pid_n * group_size
    x_offset = offs_m[:, None] * out_features + group_start + offs_n[None, :]

    # Load data from input
    x = tl.load(x_ptr + x_offset, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Compute mean and variance for the group
    mean = tl.load(mean_ptr + pid_n, mask=pid_n < num_groups, other=0.0)
    var = tl.load(var_ptr + pid_n, mask=pid_n < num_groups, other=0.0)

    # Normalize
    x = (x - mean[None, :]) / (tl.sqrt(var[None, :]) + 1e-6)

    # Scale and shift
    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    x = x * weight[None, :] + bias[None, :]

    # Store output
    tl.store(out_ptr + x_offset, x, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_add_bias(x, w, b, bias_shape, in_features, out_features, num_groups):
    # Ensure contiguous tensors
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Grid dimensions
    num_blocks_m = triton.cdiv(x.size(0), 256)
    num_blocks_n = triton.cdiv(out_features, 64)
    num_blocks_k = triton.cdiv(in_features, 64)

    # Grid and launch config
    grid = lambda meta: (num_blocks_m, num_blocks_n, num_blocks_k)
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    # Launch kernel
    matmul_add_bias_kernel[grid](
        x, w, b, out,
        x.size(0), in_features, out_features,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        1, num_groups
    )

    return out


def triton_group_norm(x, weight, bias, num_groups, group_size):
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Calculate group size
    group_size = x.size(1) // num_groups

    # Grid dimensions
    num_blocks_m = triton.cdiv(x.size(0), 128)
    num_blocks_n = triton.cdiv(group_size, 64)

    # Grid
    grid = lambda meta: (num_blocks_m, num_blocks_n)

    # Launch kernel
    group_norm_kernel[grid](
        x, None, None, weight, bias, out,
        x.size(0), x.size(1), num_groups, group_size,
        128, 64
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.weight_norm = nn.Parameter(torch.randn(out_features))
        self.bias_norm = nn.Parameter(torch.randn(out_features))
        self.num_groups = num_groups
        self.out_features = out_features

    def forward(self, x):
        # Matmul + bias + swish
        x = triton_matmul_add_bias(x, self.weight, self.bias, self.bias.shape, 
                                   self.weight.size(1), self.weight.size(0), self.num_groups)
        # Group norm
        x = triton_group_norm(x, self.weight_norm, self.bias_norm, self.num_groups, 
                              self.out_features // self.num_groups)
        return x