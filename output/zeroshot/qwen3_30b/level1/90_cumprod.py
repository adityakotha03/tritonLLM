import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
    dim_size: tl.constexpr,
):
    # Calculate the row and column indices
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Create offsets within the block
    offsets = col_idx * block_size + tl.arange(0, block_size)
    
    # Check bounds
    mask = offsets < dim_size
    
    # Load data
    x = tl.load(x_ptr + row_idx * dim_size + offsets, mask=mask, other=1.0)
    
    # Initialize cumulative product
    cumprod = tl.load(out_ptr + row_idx * dim_size + offsets, mask=mask, other=1.0)
    
    # Perform cumulative product in-place
    for i in range(1, dim_size):
        if i == 0:
            continue
        pos = col_idx * block_size + i
        if pos >= dim_size:
            break
        mask = pos < dim_size
        val = tl.load(x_ptr + row_idx * dim_size + pos, mask=mask, other=1.0)
        cumprod = cumprod * val
        tl.store(out_ptr + row_idx * dim_size + pos, cumprod, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Determine output shape and sizes
    batch_size = x.size(0)
    dim_size = x.size(dim)

    # Prepare output tensor
    out = torch.empty_like(x)

    # Define block size and grid
    BLOCK_SIZE = 128
    num_blocks = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Grid configuration: one block per row and per block of dimension
    grid = lambda meta: (batch_size, num_blocks)

    # Launch kernel
    cumprod_kernel[grid](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE, dim_size=dim_size)

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)