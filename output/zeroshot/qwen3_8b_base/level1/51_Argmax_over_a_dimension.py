import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input
    dim: tl.constexpr,  # Dimension to compute argmax over
    BLOCK_SIZE: tl.constexpr,
):
    # Determine the size of the dimension we are computing argmax over
    dim_size = tl.load(tl.make_block_ptr(x_ptr, (n_elements,), (1,), (0, 0), (0, 0), (0, 0), (0, 0)))
    # Compute the offset for the current block
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(tl.make_block_ptr(x_ptr, (n_elements,), (1,), (0, 0), (0, 0), (0, 0), (0, 0)), mask=mask, other=-float('inf'))
    # Compute the argmax along the specified dimension
    # For simplicity, we assume the dimension is the last one (dim2)
    # This is a simplified version and may need to be adjusted for other dimensions
    # In a full implementation, we would need to handle the dimension correctly
    # Here, we assume the dimension is the last dimension for demonstration
    # Compute the argmax for the current block
    max_val = tl.max(x)
    max_indices = tl.where(x == max_val, tl.arange(0, BLOCK_SIZE), -1)
    # Store the result
    tl.store(tl.make_block_ptr(out_ptr, (n_elements,), (1,), (0, 0), (0, 0), (0, 0), (0, 0)), max_indices, mask=mask)


def triton_argmax(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty(x.size(0), x.size(1), device=x.device, dtype=torch.int64)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    argmax_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our Triton-based argmax
        return triton_argmax(x, self.dim)