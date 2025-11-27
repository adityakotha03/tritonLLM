import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    alpha,  # HardSigmoid alpha (0.2)
    beta,  # HardSigmoid beta (0.5)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block start index
    xoffset = tl.program_id(0) * BLOCK_SIZE
    # Create a contiguous range of offsets for the block
    xoffsets = xoffset + tl.arange(0, BLOCK_SIZE)
    # Boolean mask for out-of-bounds elements
    xmask = xoffsets < n_elements
    # Load input value
    x = tl.load(x_ptr + xoffsets, xmask, other=0.0)
    # Broadcast alpha and beta to all threads in the block
    alpha_broadcast = tl.broadcast_to(alpha, xoffsets)
    beta_broadcast = tl.broadcast_to(beta, xoffsets)
    # Compute linear part of HardSigmoid
    tmp0 = alpha_broadcast * x
    tmp1 = tmp0 + beta_broadcast
    # Clamp the result to [0, 1]
    tmp2 = tl.clamp(tmp1, 0, 1)
    # Store the final result
    tl.store(out_ptr + xoffsets, tmp2, xmask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of HardSigmoid activation for a 2D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, D) where B=batch_size, D=dim.

    Returns:
        torch.Tensor: Output tensor of the same shape with HardSigmoid applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 2, "Input tensor must be 2-dimensional."
    # Ensure contiguous layout for memory coalescing
    x = x.contiguous()
    # Allocate output tensor with same shape and dtype
    out = torch.empty_like(x)
    n_elements = x.numel()
    alpha = 0.2  # HardSigmoid alpha
    beta = 0.5   # HardSigmoid beta
    # Choose a block size that balances occupancy and latency
    BLOCK_SIZE = 128
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    hardsigmoid_kernel[grid](x, out, n_elements, alpha, beta, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a HardSigmoid activation using a Triton kernel.

    The forward pass loads a contiguous 2D tensor, applies the Triton HardSigmoid
    kernel that implements alpha*x + beta followed by a clamp to [0,1], and returns
    the result with the same shape as the input.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0):
        # Call the Triton kernel for HardSigmoid
        return triton_hardsigmoid(input_0)