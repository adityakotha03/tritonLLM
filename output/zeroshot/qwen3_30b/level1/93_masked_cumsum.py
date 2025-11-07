import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the program ID (block index)
    pid = tl.program_id(0)
    # Determine the row (batch) and column (sequence) index within the block
    row = pid // (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    col = pid % (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Offset for current row and column within the batch
    row_offset = row * BLOCK_SIZE
    col_offset = col * BLOCK_SIZE

    # Create a range of offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Generate the actual offsets in the sequence dimension
    offsets_seq = col_offset + offsets
    # Create a mask to ensure we don't go out of bounds
    mask = offsets_seq < seq_len

    # Load the current row of data for both x and mask
    x = tl.load(x_ptr + row_offset * seq_len + offsets_seq, mask=mask, other=0.0)
    m = tl.load(mask_ptr + row_offset * seq_len + offsets_seq, mask=mask, other=0.0)

    # Apply the mask to x
    x_masked = x * m

    # Compute cumulative sum along the sequence dimension
    cumsum = tl.cumsum(x_masked, axis=0)

    # Store the result back to output
    tl.store(out_ptr + row_offset * seq_len + offsets_seq, cumsum, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor):
    """
    Optimized masked cumulative sum using Triton.
    """
    assert x.is_cuda and mask.is_cuda, "Inputs must be on CUDA."
    assert x.shape == mask.shape, "x and mask must have the same shape."

    batch_size, seq_len = x.shape
    out = torch.empty_like(x)

    # Use BLOCK_SIZE = 256 for optimal performance on A100
    BLOCK_SIZE = 256

    # Compute grid size: number of blocks needed
    grid_size = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (grid_size,)

    # Launch kernel
    masked_cumsum_kernel[grid](x, mask, out, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        # Ensure input tensors are on the same device and contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        # Apply Triton-optimized masked cumulative sum
        return triton_masked_cumsum(x, mask)