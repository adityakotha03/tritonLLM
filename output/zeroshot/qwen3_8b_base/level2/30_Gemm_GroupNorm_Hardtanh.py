import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, out_ptr,
    m, n, k,
    stride_a, stride_b, stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the row index of the output matrix
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the column index of the output matrix
    col = tl.arange(0, BLOCK_SIZE)
    # Compute the offset in the A matrix
    a_offsets = row[:, None] * stride_a + col[None, :]
    # Compute the offset in the B matrix
    b_offsets = col[None, :] * stride_b + tl.arange(0, k)
    # Load A and B
    a = tl.load(a_offsets, mask=(row < m) & (col < n), other=0.0)
    b = tl.load(b_offsets, mask=(col < k) & (tl.arange(0, k) < k), other=0.0)
    # Compute the dot product
    acc = tl.dot(a, b)
    # Store the result
    out_offsets = row[:, None] * stride_out + col[None, :]
    tl.store(out_offsets, acc, mask=(row < m) & (col < n))


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    m, n, num_groups,
    group_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Compute the row index
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the column index
    col = tl.arange(0, group_size)
    # Compute the offset in the input
    x_offsets = row[:, None] * n + col[None, :]
    # Compute the offset in gamma and beta
    gamma_offsets = col[None, :]
    beta_offsets = gamma_offsets
    # Load x, gamma, beta
    x = tl.load(x_offsets, mask=(row < m) & (col < group_size), other=0.0)
    gamma = tl.load(gamma_offsets, mask=(col < num_groups), other=1.0)
    beta = tl.load(beta_offsets, mask=(col < num_groups), other=0.0)
    # Compute mean and variance
    mean = tl.sum(x, axis=1) / group_size
    var = tl.sum((x - mean[None, :]) ** 2, axis=1) / group_size
    # Normalize
    x_norm = (x - mean[None, :]) / tl.sqrt(var[None, :] + eps)
    # Scale and shift
    out = gamma[None, :] * x_norm + beta[None, :]
    # Store the result
    out_offsets = row[:, None] * n + col[None, :]
    tl.store(out_offsets, out, mask=(row < m) & (col < group_size))


@triton.jit
def hardtanh_kernel(
    x_ptr, out_ptr,
    m, n,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Compute the row index
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the column index
    col = tl.arange(0, n)
    # Compute the offset in the input and output
    x_offsets = row[:, None] * n + col[None, :]
    # Load x
    x = tl.load(x_offsets, mask=(row < m) & (col < n), other=0.0)
    # Apply HardTanh
    x = tl.where(x < min_val, min_val, x)
    x = tl.where(x > max_val, max_val, x)
    # Store the result
    tl.store(x_offsets, x, mask=(row < m) & (col < n))


def triton_gemm(a, b, m, n, k, stride_a, stride_b, stride_out, block_size):
    # Ensure the inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()
    # Prepare output tensor
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    # Launch the kernel
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gemm_kernel[grid](a, b, out, m, n, k, stride_a, stride_b, stride_out, BLOCK_SIZE=block_size)
    return out


def triton_group_norm(x, gamma, beta, m, n, num_groups, group_size, eps, block_size):
    # Ensure the inputs are contiguous
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Launch the kernel
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    group_norm_kernel[grid](x, gamma, beta, out, m, n, num_groups, group_size, eps, BLOCK_SIZE=block_size)
    return out


def triton_hardtanh(x, m, n, min_val, max_val, block_size):
    # Ensure the input is contiguous
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Launch the kernel
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardtanh_kernel[grid](x, out, m, n, min_val, max_val, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.register_buffer('gamma', torch.ones(out_features))
        self.register_buffer('beta', torch.zeros(out_features))

    def forward(self, x):
        # GEMM
        x = triton_gemm(x, torch.randn(self.in_features, self.out_features, device=x.device), x.size(0), self.out_features, self.in_features,
                        x.stride(0), self.in_features, self.out_features, 128)
        # GroupNorm
        x = triton_group_norm(x, self.gamma, self.beta, x.size(0), self.out_features, self.num_groups, self.out_features // self.num_groups, 1e-5, 128)
        # HardTanh
        x = triton_hardtanh(x, x.size(0), self.out_features, self.hardtanh_min, self.hardtanh_max, 128)
        return x