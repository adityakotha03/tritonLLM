import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr,  # Pointer to first matrix (A)
    b_ptr,  # Pointer to second matrix (B)
    out_ptr,  # Pointer to output matrix (C)
    m,  # Number of rows in A and C
    n,  # Number of columns in B and C
    k,  # Number of columns in A and rows in B
    stride_am,  # Stride of A matrix
    stride_ab,  # Stride of B matrix
    stride_cm,  # Stride of C matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the block dimensions
    block_m = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Compute the block start
    block_start_m = pid * BLOCK_SIZE
    # Compute the block offset
    block_offset = block_start_m * stride_cm
    # Compute the block's row indices
    row_offsets = block_start_m + tl.arange(0, BLOCK_SIZE)
    # Compute the block's column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Iterate over the columns of B
    for i in range(0, k, BLOCK_SIZE):
        # Compute the offset for A and B
        a_offset = block_offset + row_offsets[:, None] * stride_am + i * stride_ab
        b_offset = block_offset + i * stride_ab + col_offsets[None, :]
        # Load A and B
        a = tl.load(a_offset, mask=(row_offsets < m)[:, None] & (col_offsets < k), other=0.0)
        b = tl.load(b_offset, mask=(row_offsets < m)[:, None] & (col_offsets < k), other=0.0)
        # Compute the dot product
        acc += tl.dot(a, b)
    # Store the result
    tl.store(out_ptr + block_offset + row_offsets[:, None] * stride_cm + col_offsets[None, :], acc, mask=(row_offsets < m)[:, None] & (col_offsets < n))


@triton.jit
def batchnorm_kernel(
    x_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    gamma_ptr,  # Pointer to gamma tensor
    eps,  # Epsilon for numerical stability
    out_ptr,  # Pointer to output tensor
    m,  # Number of rows in input
    n,  # Number of columns in input
    stride_x,  # Stride of input tensor
    stride_out,  # Stride of output tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the block dimensions
    block_m = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Compute the block start
    block_start_m = pid * BLOCK_SIZE
    # Compute the block offset
    block_offset = block_start_m * stride_x
    # Compute the block's row indices
    row_offsets = block_start_m + tl.arange(0, BLOCK_SIZE)
    # Compute the block's column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Load mean and variance
    mean = tl.load(mean_ptr + pid, other=0.0)
    var = tl.load(var_ptr + pid, other=0.0)
    # Initialize the output
    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Iterate over the columns of input
    for i in range(0, n, BLOCK_SIZE):
        # Compute the offset for input
        x_offset = block_offset + row_offsets[:, None] * stride_x + i * stride_x
        # Load input
        x = tl.load(x_offset, mask=(row_offsets < m)[:, None] & (col_offsets < n), other=0.0)
        # Compute normalization
        x_normalized = (x - mean) / tl.sqrt(var + eps)
        # Compute scaling
        out = x_normalized * tl.load(gamma_ptr + pid, other=1.0)
        # Store the result
        tl.store(out_ptr + block_offset + row_offsets[:, None] * stride_out + i * stride_out + col_offsets[None, :], out, mask=(row_offsets < m)[:, None] & (col_offsets < n))


@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    m,  # Number of rows in input
    n,  # Number of columns in input
    stride_x,  # Stride of input tensor
    stride_out,  # Stride of output tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the block dimensions
    block_m = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Compute the block start
    block_start_m = pid * BLOCK_SIZE
    # Compute the block offset
    block_offset = block_start_m * stride_x
    # Compute the block's row indices
    row_offsets = block_start_m + tl.arange(0, BLOCK_SIZE)
    # Compute the block's column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Initialize the output
    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Iterate over the columns of input
    for i in range(0, n, BLOCK_SIZE):
        # Compute the offset for input
        x_offset = block_offset + row_offsets[:, None] * stride_x + i * stride_x
        # Load input
        x = tl.load(x_offset, mask=(row_offsets < m)[:, None] & (col_offsets < n), other=0.0)
        # Compute max and exp
        max_val = tl.max(x, axis=1)
        exp_x = tl.exp(x - max_val[:, None])
        # Compute sum of exp
        sum_exp = tl.sum(exp_x, axis=1)
        # Compute softmax
        out = exp_x / sum_exp[:, None]
        # Store the result
        tl.store(out_ptr + block_offset + row_offsets[:, None] * stride_out + i * stride_out + col_offsets[None, :], out, mask=(row_offsets < m)[:, None] & (col_offsets < n))


def triton_gemm(a: torch.Tensor, b: torch.Tensor, m: int, n: int, k: int, stride_am: int, stride_ab: int, stride_cm: int, block_size: int):
    # Prepare output tensor
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    # Launch the Triton kernel
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gemm_kernel[grid](a, b, out, m, n, k, stride_am, stride_ab, stride_cm, BLOCK_SIZE=block_size)
    return out


def triton_batchnorm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, eps: float, m: int, n: int, stride_x: int, stride_out: int, block_size: int):
    # Prepare output tensor
    out = torch.empty_like(x)
    # Launch the Triton kernel
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    batchnorm_kernel[grid](x, mean, var, gamma, eps, out, m, n, stride_x, stride_out, BLOCK_SIZE=block_size)
    return out


def triton_softmax(x: torch.Tensor, m: int, n: int, stride_x: int, stride_out: int, block_size: int):
    # Prepare output tensor
    out = torch.empty_like(x)
    # Launch the Triton kernel
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softmax_kernel[grid](x, out, m, n, stride_x, stride_out, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.scale_shape = scale_shape
        self.register_buffer('scale', torch.ones(scale_shape))
        self.register_buffer('mean', torch.zeros(out_features))
        self.register_buffer('var', torch.ones(out_features))

    def forward(self, x):
        # GEMM
        a = x
        b = torch.randn(self.in_features, self.out_features, device=x.device, dtype=x.dtype)
        # Compute GEMM
        out_gemm = triton_gemm(a, b, x.size(0), self.out_features, self.in_features, a.stride(0), b.stride(0), out_gemm.stride(0), 128)
        # BatchNorm
        out_bn = triton_batchnorm(out_gemm, self.mean, self.var, self.scale, self.bn_eps, x.size(0), self.out_features, out_gemm.stride(0), out_bn.stride(0), 128)
        # Softmax
        out_softmax = triton_softmax(out_bn, x.size(0), self.out_features, out_bn.stride(0), out_softmax.stride(0), 128)
        return out_softmax