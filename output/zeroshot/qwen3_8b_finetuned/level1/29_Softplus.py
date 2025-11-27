import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Generate offset range for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offsets < n_elements
    # Load input value
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute exp(x)
    exp_x = tl.math.exp(x)
    # Add 1.0
    temp = exp_x + 1.0
    # Compute natural logarithm
    result = tl.math.log(temp)
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor):
    """
    Triton-based implementation of Softplus activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable block size (power of 2)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation using a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)