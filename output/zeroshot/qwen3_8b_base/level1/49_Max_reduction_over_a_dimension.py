import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input
    dim: tl.constexpr,  # Dimension to reduce over
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index of the current program in the block
    pid = tl.program_id(0)
    # Compute the offset for this block
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the stride for the dimension we are reducing over
    stride = tl.prod(tl.arange(0, dim))  # stride = product of dimensions before dim
    # Compute the number of elements in the reduced dimension
    num_reduced = tl.prod(tl.arange(dim + 1, x_ptr.shape[0]))  # product of dimensions after dim

    # Compute the index in the reduced dimension
    reduced_idx = (offsets // stride) % num_reduced
    # Compute the index in the non-reduced dimensions
    non_reduced_idx = offsets % stride

    # Load the input values
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=-float('inf'))

    # Compute the maximum along the reduced dimension
    max_val = tl.max(x, axis=0)

    # Store the result
    tl.store(out_ptr + reduced_idx + non_reduced_idx, max_val)


def triton_max_reduction(x: torch.Tensor, dim: int):
    """
    Applies max reduction over the specified dimension using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Compute the output shape
    output_shape = list(x.shape)
    output_shape.pop(dim)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    max_reduction_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduction(x, self.dim)