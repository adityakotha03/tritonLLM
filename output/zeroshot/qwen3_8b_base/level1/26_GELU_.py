import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    # Compute the block start and end indices
    block_start = block_idx * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    # Create offset range
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to handle boundaries
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute GELU using approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Use Taylor approximation for better performance
    x = x * (1.0 + tl.math.erf(x * 0.7071)) * 0.5
    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies GELU activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)