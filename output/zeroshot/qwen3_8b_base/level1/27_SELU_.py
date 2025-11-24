import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply SELU activation: out = alpha * (x * beta + 1) * (x >= 0) + alpha * exp(x - 1) * (x < 0)
    alpha = 1.673286795139986
    beta = 1.0507009873584274
    positive = x >= 0
    negative = x < 0
    out = tl.where(positive, alpha * (x * beta + 1), alpha * tl.exp(x - 1))
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Applies SELU activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    # Choose a block size that's a power of two
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)