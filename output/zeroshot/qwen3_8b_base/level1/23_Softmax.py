import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the tensor
    dim,  # Dimension to apply softmax
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # Compute the max value in the specified dimension
    max_val = tl.max(x, axis=dim)
    max_val = tl.broadcast_to(max_val, x.shape)

    # Subtract the max value
    x -= max_val

    # Compute the exponential
    exp_x = tl.math.exp(x)

    # Compute the sum along the specified dimension
    sum_exp = tl.sum(exp_x, axis=dim)
    sum_exp = tl.broadcast_to(sum_exp, x.shape)

    # Compute the softmax
    out = exp_x / sum_exp

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int):
    """
    Applies softmax using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softmax_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x, dim=1)