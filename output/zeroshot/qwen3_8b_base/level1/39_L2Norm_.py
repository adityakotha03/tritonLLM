import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    dim: tl.constexpr,  # Dimension to normalize
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute the norm along the specified dimension
    # We use a reduction across the dim dimension
    # For each row, compute the squared sum
    # We'll use a tile-based approach for efficiency

    # Compute the number of rows
    num_rows = n_elements // dim

    # Each thread block handles a block of rows
    row_start = pid * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < num_rows

    # Load the data for the current block of rows
    x = tl.load(x_ptr + row_offsets * dim + tl.arange(0, dim), mask=row_mask[:, None] & (tl.arange(0, dim) < dim), other=0.0)
    # Compute squared sum
    squared_sum = tl.sum(x * x, axis=1)
    # Compute norm
    norm = tl.sqrt(tl.max(squared_sum, axis=0))
    # Normalize
    out = x / norm
    # Store the result
    tl.store(out_ptr + row_offsets * dim + tl.arange(0, dim), out, mask=row_mask[:, None] & (tl.arange(0, dim) < dim))


def triton_l2norm(x: torch.Tensor, dim: int):
    """
    Applies L2 normalization using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    num_rows = n_elements // dim
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((num_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    l2norm_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2norm(x, self.dim)