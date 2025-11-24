import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,        # pointer to input tensor
    out_ptr,      # pointer to output tensor
    n_cols,       # number of columns (feature dimension)
    n_rows,       # number of rows (batch size)
    row_stride,   # stride between rows
    BLOCK_SIZE: tl.constexpr,
):
    # Compute row index and column block index
    row = tl.program_id(0)
    col_block_start = tl.program_id(1) * BLOCK_SIZE
    col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load data for this row and block
    x_ptrs = x_ptr + row * row_stride + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute sum of absolute values in this block
    abs_x = tl.abs(x)
    block_sum = tl.sum(abs_x, axis=0)

    # Reduce across blocks to get full L1 norm for the row
    row_sum = tl.sum(block_sum)

    # Avoid division by zero
    row_sum = tl.maximum(row_sum, 1e-12)

    # Normalize and write back
    norm_ptrs = out_ptr + row * row_stride + col_offsets
    tl.store(norm_ptrs, x / row_sum, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized version of L1 normalization using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."
        x = x.contiguous()
        out = torch.empty_like(x)
        n_rows, n_cols = x.shape
        row_stride = x.stride(0)

        # Choose block size
        BLOCK_SIZE = 1024
        while BLOCK_SIZE > n_cols:
            BLOCK_SIZE //= 2
        if BLOCK_SIZE < 128:
            BLOCK_SIZE = 128  # minimum reasonable block size

        # Grid: one block per row and per block of columns
        grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))

        # Launch kernel
        l1_norm_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            n_cols=n_cols,
            n_rows=n_rows,
            row_stride=row_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out