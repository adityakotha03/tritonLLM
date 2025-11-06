import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of rows
    row_id = tl.program_id(0)
    row_start = row_id * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows

    # Load the row data
    x = tl.load(x_ptr + offsets[:, None] * n_cols + tl.arange(0, n_cols)[None, :], mask=mask[:, None], other=0.0)

    # Compute L2 norm online: square, sum, then square root
    # Using fused reduction to avoid storing intermediate squared values
    x_squared = x * x
    x_squared_sum = tl.sum(x_squared, axis=1)

    # Compute L2 norm
    x_norm = tl.sqrt(x_squared_sum)

    # Avoid division by zero
    x_norm = tl.where(x_norm == 0.0, 1.0, x_norm)

    # Normalize: divide by L2 norm
    out = x / x_norm[:, None]

    # Store the result
    tl.store(out_ptr + offsets[:, None] * n_cols + tl.arange(0, n_cols)[None, :], out, mask=mask[:, None])


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 normalization using a custom Triton kernel.
    Performs online L2 normalization to reduce memory traffic.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    # Use BLOCK_SIZE = 256 for good occupancy and memory coalescing
    BLOCK_SIZE = 256

    # Grid configuration: one block per row segment
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)

    # Launch kernel
    l2_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)