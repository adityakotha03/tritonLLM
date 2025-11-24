import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the tensor
    dim: tl.constexpr,  # Dimension along which to compute reverse cumulative sum
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute reverse cumulative sum
    # We need to compute sum from the end towards the start
    # So we reverse the block and compute the cumulative sum
    # Then reverse it back
    rev_offsets = tl.arange(0, BLOCK_SIZE)[::-1]
    rev_x = tl.load(x_ptr + rev_offsets + block_start, mask=rev_offsets < n_elements, other=0.0)
    rev_cumsum = tl.cumsum(rev_x, axis=0)
    cumsum = rev_cumsum[::-1]

    # Store the result
    tl.store(out_ptr + offsets, cumsum, mask=mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Perform reverse cumulative sum along the specified dimension using a Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    reverse_cumsum_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension,
    optimized with a custom Triton kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)