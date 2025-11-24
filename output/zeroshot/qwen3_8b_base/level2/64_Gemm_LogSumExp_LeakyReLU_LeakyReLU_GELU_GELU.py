import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of the output matrix
    pid = tl.program_id(0)
    # Compute the block offset in the output matrix
    block_row = pid // (n // BLOCK_SIZE)
    block_col = pid % (n // BLOCK_SIZE)
    # Compute the row and column indices for the current block
    row = block_row * BLOCK_SIZE
    col = block_col * BLOCK_SIZE
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Iterate over the columns of the first matrix
    for i in range(0, k, BLOCK_SIZE):
        # Load the current block of the first matrix
        a = tl.load(a_ptr + row + i * stride_a, (BLOCK_SIZE, BLOCK_SIZE), mask=(i + BLOCK_SIZE <= k))
        # Iterate over the rows of the second matrix
        for j in range(0, n, BLOCK_SIZE):
            # Load the current block of the second matrix
            b = tl.load(b_ptr + j * stride_b, (BLOCK_SIZE, BLOCK_SIZE), mask=(j + BLOCK_SIZE <= n))
            # Compute the matrix multiplication
            acc += tl.dot(a, b)
    # Store the result
    tl.store(c_ptr + row + col * stride_c, acc, mask=(row + BLOCK_SIZE <= m) & (col + BLOCK_SIZE <= n))


@triton.jit
def leaky_relu_kernel(
    x_ptr, out_ptr,
    m, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    mask = (row + BLOCK_SIZE) < m
    offsets = row + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, x * 0.01)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    m, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    mask = (row + BLOCK_SIZE) < m
    offsets = row + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Approximate GELU using erf
    out = 0.5 * x * (1.0 + tl.math.erf(x / tl.math.sqrt(2.0)))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    m, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    mask = (row + BLOCK_SIZE) < m
    offsets = row + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    max_val = tl.max(x, axis=0)
    exp_vals = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    out = max_val + tl.math.log(sum_exp)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    m = x.shape[0]
    n = weight.shape[1]
    k = x.shape[1]
    stride_a = x.stride(0)
    stride_b = weight.stride(0)
    stride_c = torch.zeros(m, n).stride(0)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    num_blocks = (n + 128 - 1) // 128
    grid = lambda meta: (num_blocks,)
    gemm_relu_kernel[grid](x, weight, out, m, n, k, stride_a, stride_b, stride_c, BLOCK_SIZE=128)
    if bias is not None:
        out += bias
    return out


def triton_leaky_relu(x: torch.Tensor):
    m, n = x.shape
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    num_blocks = (m + 128 - 1) // 128
    grid = lambda meta: (num_blocks,)
    leaky_relu_kernel[grid](x, out, m, n, BLOCK_SIZE=128)
    return out


def triton_gelu(x: torch.Tensor):
    m, n = x.shape
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    num_blocks = (m + 128 - 1) // 128
    grid = lambda meta: (num_blocks,)
    gelu_kernel[grid](x, out, m, n, BLOCK_SIZE=128)
    return out


def triton_logsumexp(x: torch.Tensor, dim: int):
    m, n = x.shape
    out = torch.empty((m, 1), device=x.device, dtype=x.dtype)
    num_blocks = (m + 128 - 1) // 128
    grid = lambda meta: (num_blocks,)
    logsumexp_kernel[grid](x, out, m, n, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x):
        # Gemm
        x = triton_gemm(x, self.weight, self.bias)
        # LogSumExp
        x = triton_logsumexp(x, dim=1)
        # LeakyReLU
        x = triton_leaky_relu(x)
        # LeakyReLU
        x = triton_leaky_relu(x)
        # GELU
        x = triton_gelu(x)
        # GELU
        x = triton_gelu(x)
        return x