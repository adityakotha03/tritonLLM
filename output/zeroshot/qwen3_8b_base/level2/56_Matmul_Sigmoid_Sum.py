import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_sigmoid_sum_kernel(
    a_ptr, b_ptr, out_ptr,
    n_rows, n_cols, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the row index
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds
    mask = row < n_rows
    # Load matrix A
    a = tl.load(a_ptr + row[:, None] * k + tl.arange(0, k), mask=mask[:, None] * (tl.arange(0, k) < k), other=0.0)
    # Compute the matrix multiplication
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(k):
        a_col = tl.load(b_ptr + i * n_cols + tl.arange(0, n_cols), mask=(tl.arange(0, n_cols) < n_cols), other=0.0)
        acc += tl.dot(a, a_col)
    # Apply sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))
    # Store the result
    tl.store(out_ptr + row, acc, mask=mask)


def triton_matmul_sigmoid_sum(a, b):
    """
    Perform matrix multiplication, apply sigmoid, and sum the result.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    n_rows = a.shape[0]
    n_cols = b.shape[1]
    k = a.shape[1]
    # Output shape is (n_rows, 1)
    out = torch.empty((n_rows, 1), device=a.device, dtype=a.dtype)
    # Choose block size (power of 2)
    BLOCK_SIZE = 128
    # Compute the grid size
    grid = lambda meta: ((n_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the kernel
    matmul_sigmoid_sum_kernel[grid](a, b, out, n_rows, n_cols, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x = triton_matmul_sigmoid_sum(x, torch.ones((x.shape[1], x.shape[1]), device=x.device))
        x = torch.sum(x, dim=1, keepdim=True)
        return x