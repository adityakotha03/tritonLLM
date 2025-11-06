import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch along dim1
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Initialize output index to 0 for each row
    # We use a 32-bit integer for indices
    min_indices = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    min_values = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)

    # Load data in chunks to handle large dim2
    for start in range(0, dim2, BLOCK_SIZE):
        # Compute actual block size for this iteration
        curr_block_size = min(BLOCK_SIZE, dim2 - start)
        block_offsets = start + tl.arange(0, curr_block_size)

        # Load data for current block
        x = tl.load(x_ptr + pid * dim1 * dim2 + block_offsets, mask=block_offsets < dim2, other=float('inf'))
        x = tl.load(x_ptr + pid * dim1 * dim2 + block_offsets, mask=block_offsets < dim2, other=float('inf'))

        # We need to compare values across the dim2 dimension, so we reduce over dim2
        # Initialize indices for this block
        block_indices = tl.arange(0, curr_block_size) + start
        # Check which values are smaller
        mask = x < min_values
        min_values = tl.where(mask, x, min_values)
        min_indices = tl.where(mask, block_indices, min_indices)

    # Store the final argmin index for this row
    tl.store(out_ptr + pid * dim1, min_indices, mask=offsets < dim1)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Custom Triton implementation of argmin along dim1 for a fixed dim2 = 4095.
    Optimized for A100 with specific batch size and dimensions.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    # Expected shape: (batch_size, dim1, dim2) = (128, 4096, 4095)
    batch_size = x.size(0)
    dim1 = x.size(1)
    dim2 = x.size(2)

    # Ensure dim is 1 (as in the given example)
    assert dim == 1, "Only dim=1 is supported"

    # Prepare output tensor
    out = torch.empty((batch_size, dim1), dtype=torch.int32, device=x.device)

    # Tune BLOCK_SIZE for optimal performance
    BLOCK_SIZE = 128  # Chosen for A100: 128 is a good balance for shared memory and occupancy

    # Grid: one block per row in dim1
    grid = lambda meta: (batch_size * dim1 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    # Launch the kernel
    argmin_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)