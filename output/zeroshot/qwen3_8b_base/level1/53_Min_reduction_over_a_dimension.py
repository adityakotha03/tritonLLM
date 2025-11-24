import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input tensor
    dim,  # Dimension to reduce over
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index of the current block
    pid = tl.program_id(0)
    # Compute the offset for the current block
    block_start = pid * BLOCK_SIZE
    # Compute the range of indices for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask, other=tl.max_float)

    # Compute the minimum value along the specified dimension
    # We assume the input is contiguous and the dimension is the last one
    # So we can treat it as a 1D array and reduce along the last dimension
    # This is a simplified version for demonstration; for a general case, more complex logic is needed
    # For a real implementation, the kernel should handle arbitrary dimensions and strides
    # Here, we assume that the dimension is the last one and the tensor is contiguous
    min_val = tl.min(x)

    # Store the result
    tl.store(out_ptr + pid, min_val, mask=pid < n_elements // BLOCK_SIZE)


def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Compute the output shape
    # Assuming the dimension is the last one and the tensor is contiguous
    # For a general case, the output shape should be computed based on the dimension
    # Here, we assume the dimension is the last one for simplicity
    output_shape = list(x.shape)
    output_shape[dim] = 1
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = x.numel()
    # Determine the number of blocks needed
    # For simplicity, we assume the dimension is the last one
    # and the output is a single value per block
    num_blocks = (n_elements // BLOCK_SIZE) + (1 if n_elements % BLOCK_SIZE != 0 else 0)

    # Launch the Triton kernel
    min_reduction_kernel[ num_blocks ](x, output, n_elements, dim, BLOCK_SIZE=1024)
    return output


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min_reduction(x, self.dim)