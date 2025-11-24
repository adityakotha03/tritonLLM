import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    dim: tl.constexpr,  # Dimension along which to compute cumulative sum
    size: tl.constexpr,  # Size of the tensor along the dim
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index along the dim
    pid = tl.program_id(0)
    # Compute the offset in the dim dimension
    offset = pid * BLOCK_SIZE
    # Create a range of offsets in the other dimensions
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the indices for the current block
    idx = offset + offsets
    # Mask to ensure we don't go out of bounds
    mask = idx < size
    # Load input values
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    # Compute exclusive cumulative sum
    # We need to compute the cumulative sum without including the current element
    # So we shift the input and compute the cumulative sum
    # We can do this by using the previous cumulative sum as the current value
    # We need to handle the first element specially
    # For simplicity, we assume the input is contiguous and the dim is 0
    # We'll compute the cumulative sum along the dim
    # For the first element, we set it to 0
    # For the rest, we add the previous value to the current value
    # We'll use a loop to compute the cumulative sum
    # We'll use a register to store the previous value
    prev = 0.0
    for i in range(BLOCK_SIZE):
        if mask[i]:
            prev = tl.load(x_ptr + idx[i], mask=mask[i], other=0.0)
            if i > 0:
                prev += tl.load(x_ptr + idx[i-1], mask=mask[i-1], other=0.0)
            tl.store(out_ptr + idx[i], prev, mask=mask[i])
        else:
            break


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Get the size along the dim
    size = x.size(dim)
    # Choose a block size that is a power of two
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    exclusive_cumsum_kernel[triton.make_kernel(num_blocks, 1)](x, out, dim, size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Call our Triton-based exclusive cumulative sum
        return triton_exclusive_cumsum(x, self.dim)