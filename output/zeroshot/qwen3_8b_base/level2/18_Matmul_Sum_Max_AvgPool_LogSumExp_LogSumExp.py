import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    a_rows, a_cols, b_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the row and column indices for this block
    row_start = pid * BLOCK_SIZE
    row_end = tl.minimum(row_start + BLOCK_SIZE, a_rows)
    col_start = 0
    col_end = b_cols

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the columns of B
    for col in range(col_start, col_end, BLOCK_SIZE):
        # Load the B matrix block
        b_block = tl.load(b_ptr + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * b_cols + tl.arange(0, BLOCK_SIZE), mask=col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < b_cols, other=0.0)

        # Iterate over the rows of A
        for row in range(row_start, row_end, BLOCK_SIZE):
            a_block = tl.load(a_ptr + row * a_cols + tl.arange(0, BLOCK_SIZE) * a_cols + tl.arange(0, BLOCK_SIZE), mask=row * a_cols + tl.arange(0, BLOCK_SIZE) < a_cols, other=0.0)
            # Compute the dot product
            acc += tl.dot(a_block, b_block)

    # Write the result
    tl.store(c_ptr + row_start * b_cols + tl.arange(0, BLOCK_SIZE) * b_cols + tl.arange(0, BLOCK_SIZE), acc, mask=row_start * b_cols + tl.arange(0, BLOCK_SIZE) * b_cols + tl.arange(0, BLOCK_SIZE) < a_rows * b_cols)


@triton.jit
def sum_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, out)


@triton.jit
def max_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    out = tl.max(x, axis=0)
    tl.store(out_ptr + pid, out)


@triton.jit
def mean_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.sum(x, axis=0) / BLOCK_SIZE
    tl.store(out_ptr + pid, out)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    out = max_val + tl.math.log(sum_exp)
    tl.store(out_ptr + pid, out)


def triton_matmul(a, b):
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape
    c = torch.empty((a_rows, b_cols), device=a.device, dtype=a.dtype)
    num_blocks = (a_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    matmul_kernel[grid](a, b, c, a_rows, a_cols, b_cols, BLOCK_SIZE=1024)
    return c


def triton_sum(x):
    n_elements = x.numel()
    out = torch.empty((n_elements // 1024 + (n_elements % 1024 != 0)), device=x.device, dtype=x.dtype)
    num_blocks = (n_elements + 1024 - 1) // 1024
    grid = lambda meta: (num_blocks,)
    sum_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out[0]


def triton_max(x):
    n_elements = x.numel()
    out = torch.empty((n_elements // 1024 + (n_elements % 1024 != 0)), device=x.device, dtype=x.dtype)
    num_blocks = (n_elements + 1024 - 1) // 1024
    grid = lambda meta: (num_blocks,)
    max_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out[0]


def triton_mean(x):
    n_elements = x.numel()
    out = torch.empty((n_elements // 1024 + (n_elements % 1024 != 0)), device=x.device, dtype=x.dtype)
    num_blocks = (n_elements + 1024 - 1) // 1024
    grid = lambda meta: (num_blocks,)
    mean_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out[0]


def triton_logsumexp(x):
    n_elements = x.numel()
    out = torch.empty((n_elements // 1024 + (n_elements % 1024 != 0)), device=x.device, dtype=x.dtype)
    num_blocks = (n_elements + 1024 - 1) // 1024
    grid = lambda meta: (num_blocks,)
    logsumexp_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out[0]


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = triton_sum(x)
        x = triton_max(x)
        x = triton_mean(x)
        x = triton_logsumexp(x)
        x = triton_logsumexp(x)
        return x