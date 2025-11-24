import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr,  # pointer to matrix A
    b_ptr,  # pointer to matrix B
    out_ptr,  # pointer to output matrix
    n, m, k,  # dimensions of A (n x k), B (k x m)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of the output matrix
    pid = tl.program_id(0)
    # Compute the block's row and column indices
    block_row = pid // (m // BLOCK_SIZE)
    block_col = pid % (m // BLOCK_SIZE)
    # Compute the offset in the output matrix
    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load matrix A and B
    a = tl.load(a_ptr + row_offsets[:, None] * k + tl.arange(0, k), mask=row_offsets[:, None] * k + tl.arange(0, k) < n * k, other=0.0)
    b = tl.load(b_ptr + col_offsets[None, :] * k + tl.arange(0, k), mask=col_offsets[None, :] * k + tl.arange(0, k) < k * m, other=0.0)
    # Compute the matrix multiplication
    c = tl.dot(a, b)
    # Store the result
    tl.store(out_ptr + row_offsets[:, None] * m + col_offsets[None, :], c, mask=row_offsets[:, None] * m + col_offsets[None, :] < n * m)


@triton.jit
def batch_norm_kernel(
    x_ptr,  # pointer to input matrix
    mean_ptr,  # pointer to mean
    var_ptr,  # pointer to variance
    scale_ptr,  # pointer to scale
    out_ptr,  # pointer to output matrix
    n, m,  # dimensions of input matrix (n x m)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of the input matrix
    pid = tl.program_id(0)
    # Compute the block's row and column indices
    block_row = pid // (m // BLOCK_SIZE)
    block_col = pid % (m // BLOCK_SIZE)
    # Compute the offset in the input matrix
    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load input matrix
    x = tl.load(x_ptr + row_offsets[:, None] * m + col_offsets[None, :], mask=row_offsets[:, None] * m + col_offsets[None, :] < n * m, other=0.0)
    # Compute mean and variance
    mean = tl.sum(x, axis=1) / n
    var = tl.sum((x - mean[None, :]) ** 2, axis=1) / n
    # Store mean and variance
    tl.store(mean_ptr + block_row, mean, mask=block_row < n)
    tl.store(var_ptr + block_row, var, mask=block_row < n)
    # Normalize and scale
    x_hat = (x - mean[None, :]) / tl.sqrt(var[None, :] + eps)
    out = x_hat * tl.load(scale_ptr + block_row, mask=block_row < n)
    # Store the result
    tl.store(out_ptr + row_offsets[:, None] * m + col_offsets[None, :], out, mask=row_offsets[:, None] * m + col_offsets[None, :] < n * m)


def triton_gemm(a: torch.Tensor, b: torch.Tensor, n, m, k):
    # Ensure the inputs are contiguous on GPU
    a = a.contiguous()
    b = b.contiguous()
    # Prepare output tensor
    out = torch.empty((n, m), device=a.device, dtype=a.dtype)
    # Determine the number of blocks needed
    BLOCK_SIZE = 128  # Tunable parameter for block size
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    gemm_kernel[triton.make_kernel(num_blocks)](a, b, out, n, m, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_batch_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, scale: torch.Tensor):
    # Ensure the inputs are contiguous on GPU
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    scale = scale.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Determine the number of blocks needed
    BLOCK_SIZE = 128  # Tunable parameter for block size
    num_blocks = (x.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    batch_norm_kernel[triton.make_kernel(num_blocks)](x, mean, var, scale, out, x.shape[0], x.shape[1], eps=1e-5, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        # Perform GEMM using Triton kernel
        x = triton_gemm(x, self.gemm.weight, x.shape[0], self.out_features, self.in_features)
        # Apply scaling
        x = x * self.scale
        # Perform batch normalization using Triton kernel
        x = triton_batch_norm(x, self.running_mean, self.running_var, self.scale)
        return x