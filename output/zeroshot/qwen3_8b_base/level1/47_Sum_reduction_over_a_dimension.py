import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input
    dim,  # Dimension to reduce over
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the offset for this program
    offset = pid * BLOCK_SIZE
    # Create a range of indices for this block
    indices = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = indices < n_elements
    # Load the input values
    x = tl.load(x_ptr + indices, mask=mask, other=0.0)
    # Compute the sum along the specified dimension
    # We assume the dimension is the last one for simplicity
    # and that the input is contiguous in the reduced dimension
    # For a general case, more complex indexing would be needed
    # Here, we assume that the dimension is the last dimension
    # and that the input is stored in row-major order
    # So, we sum across the last dimension
    sum_val = tl.sum(x, axis=0)
    # Store the result
    tl.store(out_ptr + pid, sum_val, mask=pid < n_elements // BLOCK_SIZE)


def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Compute the output shape
    output_shape = list(x.shape)
    output_shape[dim] = 1
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    # Number of elements in the tensor
    n_elements = x.numel()
    # Choose a block size (can be tuned)
    BLOCK_SIZE = 1024

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sum_reduction_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduction(x, self.dim)