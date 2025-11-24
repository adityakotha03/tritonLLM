import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input (matrix A)
    b_ptr,  # Pointer to second input (matrix B)
    out_ptr,  # Pointer to output (matrix C)
    n_rows,  # Number of rows in A and C
    n_cols,  # Number of columns in B and C
    k,  # Number of columns in A and rows in B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < n_rows

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load matrix A
    a = tl.load(a_ptr + row[:, None] * k + tl.arange(0, k), mask=mask[:, None] * (tl.arange(0, k) < k), other=0.0)

    # Iterate over columns of B
    for i in range(0, k, BLOCK_SIZE):
        col = tl.arange(0, BLOCK_SIZE)
        b = tl.load(b_ptr + col + i * n_cols, mask=(col < k) & (i + col < k), other=0.0)
        acc += tl.dot(a, b)

    # Store result
    tl.store(out_ptr + row[:, None] * n_cols + tl.arange(0, n_cols), acc, mask=mask[:, None])


@triton.jit
def avg_pool_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    pool_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Compute the start and end indices for the pooling window
    start = offset // pool_size
    end = start + 1
    window = tl.load(x_ptr + offset, mask=mask, other=0.0)
    window = window[None, :]  # Reshape to (1, BLOCK_SIZE)
    window = window.reshape((1, pool_size, -1))
    avg = tl.mean(window, axis=1)
    avg = avg.reshape(-1)
    tl.store(out_ptr + offset, avg, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    # Approximate GELU using tanh
    x = x * (1.0 + tl.tanh(0.0356774 * x * (1.0507009873554804934 * x + -1.0507009873554804934 * x ** 3)))
    tl.store(out_ptr + offset, x, mask=mask)


@triton.jit
def scale_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, x * scale, mask=mask)


@triton.jit
def max_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=-float('inf'))
    tl.store(out_ptr + offset, x, mask=mask)


def matmul(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Ensure shapes are correct
    n_rows, k = x.shape
    _, n_cols = y.shape

    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)

    # Determine block size
    BLOCK_SIZE = 128

    # Grid size
    grid = lambda meta: ((n_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    matmul_kernel[grid](x, y, out, n_rows, n_cols, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def avg_pool(x: torch.Tensor, pool_size: int):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    out = torch.empty((n_elements // pool_size), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    avg_pool_kernel[grid](x, out, n_elements, pool_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def gelu(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    out = torch.empty((n_elements), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def scale(x: torch.Tensor, scale_factor: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    out = torch.empty((n_elements), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    scale_kernel[grid](x, out, n_elements, scale_factor, BLOCK_SIZE=BLOCK_SIZE)
    return out


def max(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    out = torch.empty((n_elements), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = matmul(x, torch.randn(self.in_features, self.out_features, device=x.device))
        x = avg_pool(x, self.pool_kernel_size)
        x = gelu(x)
        x = scale(x, self.scale_factor)
        x = max(x)
        return x