import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the input tensor
    dim: tl.constexpr,  # Dimension along which to compute argmin
    BLOCK_SIZE: tl.constexpr,
):
    # Determine the offset for the current block
    pid = tl.program_id(0)
    # Compute the size of the dimension along which we are computing argmin
    dim_size = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE) * x_ptr.strides[dim] + tl.arange(0, BLOCK_SIZE) * x_ptr.strides[dim], mask=tl.arange(0, BLOCK_SIZE) < n_elements // x_ptr.strides[dim], other=0)
    # Compute the offset in the tensor for the current block
    offset = pid * BLOCK_SIZE
    # Create a range of indices for the current block
    idx = offset + tl.arange(0, BLOCK_SIZE)
    # Compute the indices along the specified dimension
    indices = idx // x_ptr.strides[dim]
    # Compute the values along the specified dimension
    values = tl.load(x_ptr + idx, mask=idx < n_elements, other=0.0)
    # Find the minimum value and its index
    min_val = tl.max(values)
    min_idx = tl.argmax(values)
    # Store the result
    tl.store(out_ptr + indices, min_idx, mask=indices < n_elements // x_ptr.strides[dim])


def triton_argmin(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out_shape = list(x.shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    argmin_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)