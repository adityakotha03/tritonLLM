import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute Softplus: softplus(x) = log(1 + exp(x))
    exp_x = tl.exp(x)
    out = tl.log(1.0 + exp_x)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Softplus activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    # Choose a suitable block size
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)