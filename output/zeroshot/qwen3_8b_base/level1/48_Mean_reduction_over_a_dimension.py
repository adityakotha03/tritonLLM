import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input
    dim,  # Dimension to reduce over
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Get the number of elements per block
    block_size = tl.num_programs(0) * BLOCK_SIZE
    # Compute the offset for this block
    offset = pid * block_size
    # Create a range of offsets [0, block_size)
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute the sum along the specified dimension
    sum_vals = tl.sum(x, axis=dim)
    # Compute the count along the specified dimension
    count = tl.numel(x) // tl.numel(x)  # Placeholder, need to handle dimension correctly
    # Compute the mean
    mean = sum_vals / count
    # Store the result
    tl.store(out_ptr + offsets, mean, mask=mask)


def triton_mean_reduction(x: torch.Tensor, dim: int):
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

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    mean_reduction_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean_reduction(x, self.dim)