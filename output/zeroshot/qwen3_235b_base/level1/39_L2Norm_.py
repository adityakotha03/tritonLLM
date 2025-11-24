import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_cols,         # Number of columns (feature dimension)
    n_rows,         # Number of rows (batch size)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute offsets for this row
    row_start = row_idx * n_cols
    col_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to avoid out-of-bounds access
    mask = col_offsets < (row_idx + 1) * n_cols
    
    # Load the row data
    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute squared sum (L2 norm squared)
    x_sq = x * x
    l2_sq = tl.sum(x_sq, axis=0)
    
    # Compute 1 / L2 norm (i.e., reciprocal of sqrt(l2_sq + eps))
    # Use tl.sqrt and handle case where norm is zero
    l2_norm_inv = tl.math.rsqrt(l2_sq + 1e-12)
    
    # Normalize and store
    out = x * l2_norm_inv
    tl.store(out_ptr + col_offsets, out, mask=mask)

def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Choose block size as power of 2 that covers n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Grid: one block per row
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    l2_norm_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)