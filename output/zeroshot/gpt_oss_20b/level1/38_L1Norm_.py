import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_abs_kernel(
    x_ptr,          # Input tensor
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    out_ptr,        # Output tensor (mean per row)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute the mean of the absolute values for each row.
    """
    row_id = tl.program_id(0)
    row_offset = row_id * n_cols

    # Accumulator for the sum of abs values
    sum_val = tl.zeros([1], dtype=tl.float32)[0]

    # Iterate over the columns in chunks of BLOCK_SIZE
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = row_offset + offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < row_offset + n_cols
        vals = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(tl.abs(vals))

    mean_val = sum_val / n_cols
    tl.store(out_ptr + row_id, mean_val)


@triton.jit
def div_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Divide each element of the input by the mean of its row.
    """
    row_id = tl.program_id(0)
    row_offset = row_id * n_cols

    # Load the mean for this row once
    mean_val = tl.load(mean_ptr + row_id)

    # Iterate over columns in chunks of BLOCK_SIZE
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = row_offset + offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < row_offset + n_cols
        vals = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
        out_vals = vals / mean_val
        tl.store(out_ptr + col_offsets, out_vals, mask=mask)


def _sum_abs_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the Triton kernel that returns a mean per row.
    """
    assert x.is_cuda
    n_rows, n_cols = x.shape
    mean = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 256  # can be tuned or autotuned

    grid = lambda meta: (n_rows,)
    sum_abs_kernel[grid](x, n_rows, n_cols, mean, BLOCK_SIZE=BLOCK_SIZE)
    return mean


def _normalize_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that performs L1 normalization using the Triton kernels.
    """
    n_rows, n_cols = x.shape
    mean = _sum_abs_torch(x)
    out = torch.empty_like(x)
    BLOCK_SIZE = 256

    grid = lambda meta: (n_rows,)
    div_kernel[grid](x, mean, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Triton‑accelerated L1‑normalization model.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _normalize_torch(x)