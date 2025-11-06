import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
):
    # Each program handles a row of the input tensor
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # Calculate the starting offset for this block
    row_offset = row * row_stride
    col_offset = col_block * BLOCK_SIZE

    # Create a range of column offsets for this block
    offsets = col_offset + tl.arange(0, BLOCK_SIZE)

    # Mask to handle boundary conditions
    mask = offsets < n_cols

    # Load the input data
    x = tl.load(x_ptr + row_offset + offsets * col_stride, mask=mask, other=-float('inf'))

    # Initialize the max value with negative infinity
    max_val = -float('inf')

    # Iterate over the columns to find the maximum value
    for i in range(BLOCK_SIZE):
        if mask[i]:
            max_val = tl.maximum(max_val, x[i])

    # Store the result in the output tensor
    tl.store(out_ptr + row * n_cols, max_val)


def triton_max_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Custom Triton implementation of max reduction over a specified dimension.
    Uses a block-based reduction with optimized memory access patterns.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."

    # Ensure the input is contiguous
    x = x.contiguous()

    # Get the shape of the input tensor
    shape = x.shape
    n_rows, n_cols = shape[0], shape[1]

    # Prepare the output tensor
    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)

    # Block size (must be power of 2)
    BLOCK_SIZE = 128

    # Determine the grid
    grid = lambda meta: (n_rows, (n_cols + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    max_reduction_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        USE_TENSOR_CORES=False,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduction(x, self.dim)