import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, out_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block size and offset
    block_m = m // (triton.cdiv(m, BLOCK_SIZE))
    block_n = n // (triton.cdiv(n, BLOCK_SIZE))
    # Compute the block start
    block_start_m = pid // block_n
    block_start_n = pid % block_n
    # Compute the block offsets
    offs_m = block_start_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = block_start_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bounds
    mask = (offs_m < m) & (offs_n < n)
    # Load A and B
    a = tl.load(a_ptr + offs_m[:, None] * k + offs_n[None, :], mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_n[None, :] * k + offs_m[:, None], mask=mask, other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(out_ptr + offs_m[:, None] * n + offs_n[None, :], c, mask=mask)


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    n_groups, c_per_group, h, w,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block size and offset
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    # Compute the offset for the current block
    offset = block_start + tl.arange(0, block_size)
    # Mask for out-of-bounds
    mask = offset < (n_groups * c_per_group * h * w)
    # Compute the group index and local index
    group_idx = (offset // (c_per_group * h * w)) % n_groups
    local_idx = offset % (c_per_group * h * w)
    # Compute the group mean and variance
    mean = tl.sum(tl.load(x_ptr + offset, mask=mask, other=0.0), axis=0) / (c_per_group * h * w)
    var = tl.sum(tl.load(x_ptr + offset, mask=mask, other=0.0) - mean, axis=0) ** 2 / (c_per_group * h * w)
    # Normalize the data
    x_norm = (tl.load(x_ptr + offset, mask=mask, other=0.0) - mean) / tl.sqrt(var + 1e-5)
    # Apply gamma and beta
    x_norm = x_norm * tl.load(gamma_ptr + group_idx * c_per_group * h * w + offset, mask=mask, other=0.0)
    x_norm = x_norm + tl.load(beta_ptr + group_idx * c_per_group * h * w + offset, mask=mask, other=0.0)
    # Store the result
    tl.store(out_ptr + offset, x_norm, mask=mask)


@triton.jit
def min_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block size and offset
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    # Compute the offset for the current block
    offset = block_start + tl.arange(0, block_size)
    # Mask for out-of-bounds
    mask = offset < n_elements
    # Load the data
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    # Compute the minimum
    min_val = tl.min(x, axis=0)
    # Store the result
    tl.store(out_ptr + offset, min_val, mask=mask)


@triton.jit
def bias_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block size and offset
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    # Compute the offset for the current block
    offset = block_start + tl.arange(0, block_size)
    # Mask for out-of-bounds
    mask = offset < n_elements
    # Load the data
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offset, mask=mask, other=0.0)
    # Add bias
    out = x + bias
    # Store the result
    tl.store(out_ptr + offset, out, mask=mask)


def triton_gemm(a, b, m, n, k, BLOCK_SIZE=128):
    # Prepare output tensor
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    # Determine the number of blocks needed
    grid = lambda meta: ((triton.cdiv(m, meta["BLOCK_SIZE"]) * triton.cdiv(n, meta["BLOCK_SIZE"])),)
    # Launch the Triton kernel
    gemm_kernel[grid](a, b, out, m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_group_norm(x, gamma, beta, n_groups, c_per_group, h, w, BLOCK_SIZE=128):
    # Prepare output tensor
    out = torch.empty_like(x)
    # Determine the number of blocks needed
    grid = lambda meta: ((triton.cdiv(n_groups * c_per_group * h * w, meta["BLOCK_SIZE"]),),)
    # Launch the Triton kernel
    group_norm_kernel[grid](x, gamma, beta, out, n_groups, c_per_group, h, w, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_min(x, n_elements, BLOCK_SIZE=128):
    # Prepare output tensor
    out = torch.empty((n_elements,), device=x.device, dtype=x.dtype)
    # Determine the number of blocks needed
    grid = lambda meta: ((triton.cdiv(n_elements, meta["BLOCK_SIZE"]),),)
    # Launch the Triton kernel
    min_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_bias_add(x, bias, n_elements, BLOCK_SIZE=128):
    # Prepare output tensor
    out = torch.empty_like(x)
    # Determine the number of blocks needed
    grid = lambda meta: ((triton.cdiv(n_elements, meta["BLOCK_SIZE"]),),)
    # Launch the Triton kernel
    bias_add_kernel[grid](x, bias, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.bias_shape = bias_shape
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # GEMM
        x = triton_gemm(x, torch.randn((self.out_features, self.in_features), device=x.device), x.size(0), self.out_features, self.in_features)
        # GroupNorm
        x = triton_group_norm(x, torch.randn((self.out_features,), device=x.device), torch.randn((self.out_features,), device=x.device), self.num_groups, self.out_features // self.num_groups, 1, 1)
        # Min
        x = triton_min(x, x.size(0) * self.out_features)
        # Bias add
        x = triton_bias_add(x, self.bias, x.size(0) * self.out_features)
        # Reshape back to original shape
        x = x.view(x.size(0), self.out_features, 1, 1)
        return x