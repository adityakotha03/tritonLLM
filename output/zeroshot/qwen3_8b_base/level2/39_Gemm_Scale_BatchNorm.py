import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_relu_kernel(
    a_ptr,  # Pointer to first input matrix (a)
    b_ptr,  # Pointer to second input matrix (b)
    out_ptr,  # Pointer to output matrix
    m,  # Number of rows in a and out
    n,  # Number of columns in b and out
    k,  # Number of columns in a and rows in b
    stride_a,  # Stride of a
    stride_b,  # Stride of b
    stride_out,  # Stride of out
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance computes a block of the output matrix
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the block offset
    block_row = pid // (n // BLOCK_SIZE)
    block_col = pid % (n // BLOCK_SIZE)
    # Compute the row and column indices in the output block
    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the offset in a and b
    a_offsets = row_offsets[:, None] * stride_a + tl.arange(0, k)[None, :]
    b_offsets = tl.arange(0, k)[:, None] + col_offsets[None, :] * stride_b
    # Load a and b
    a = tl.load(a_offsets, mask=(row_offsets < m)[:, None] & (tl.arange(0, k) < k)[None, :], other=0.0)
    b = tl.load(b_offsets, mask=(tl.arange(0, k) < k)[:, None] & (col_offsets < n)[None, :], other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Apply ReLU
    c = tl.maximum(c, 0.0)
    # Store the result
    tl.store(out_ptr + row_offsets[:, None] * stride_out + col_offsets[None, :], c, mask=(row_offsets < m)[:, None] & (col_offsets < n)[None, :])


@triton.jit
def bn_kernel(
    x_ptr,  # Pointer to input matrix
    mean_ptr,  # Pointer to mean
    var_ptr,  # Pointer to variance
    scale_ptr,  # Pointer to scale
    out_ptr,  # Pointer to output matrix
    m,  # Number of rows
    n,  # Number of columns
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance computes a block of the output matrix
    pid = tl.program_id(0)
    block_row = pid // (n // BLOCK_SIZE)
    block_col = pid % (n // BLOCK_SIZE)
    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the offset in x
    x_offsets = row_offsets[:, None] * n + col_offsets[None, :]
    # Load x
    x = tl.load(x_ptr + x_offsets, mask=(row_offsets < m)[:, None] & (col_offsets < n)[None, :], other=0.0)
    # Compute mean and variance (already computed)
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)
    # Compute batch normalization
    scale = tl.load(scale_ptr)
    x_hat = (x - mean) / tl.sqrt(var + eps)
    out = x_hat * scale
    # Store the result
    tl.store(out_ptr + x_offsets, out, mask=(row_offsets < m)[:, None] & (col_offsets < n)[None, :])


def triton_gemm_relu(a: torch.Tensor, b: torch.Tensor, m: int, n: int, k: int, stride_a: int, stride_b: int, stride_out: int):
    # Determine block size
    BLOCK_SIZE = 128
    # Compute the number of blocks needed
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    grid = lambda meta: (num_blocks,)
    gemm_relu_kernel[grid](a, b, out_ptr, m, n, k, stride_a, stride_b, stride_out, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_bn(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, scale: torch.Tensor, m: int, n: int, eps: float):
    # Determine block size
    BLOCK_SIZE = 128
    # Compute the number of blocks needed
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    grid = lambda meta: (num_blocks,)
    bn_kernel[grid](x, mean, var, scale, out_ptr, m, n, eps, BLOCK_SIZE=BLOCK_SIZE)
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
        # Perform matrix multiplication
        a = x
        b = torch.randn(self.in_features, self.out_features).cuda()
        m, n = x.size(0), self.out_features
        k = self.in_features
        stride_a = a.stride(0)
        stride_b = b.stride(0)
        stride_out = torch.zeros(m, self.out_features).cuda().stride(0)
        out = torch.zeros(m, self.out_features).cuda()
        out = triton_gemm_relu(a, b, m, n, k, stride_a, stride_b, stride_out)
        # Scale
        out = out * self.scale
        # Batch normalization
        mean = self.running_mean
        var = self.running_var
        out = triton_bn(out, mean, var, self.scale, m, n, self.eps)
        return out