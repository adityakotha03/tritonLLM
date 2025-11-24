import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh using Taylor series approximation
    # tanh(x) = x - x^3/3 + x^5/5 - x^7/7 + ... for |x| < 1
    # For |x| >= 1, use tanh(x) = sign(x) * (1 - 2 * exp(-2|x|)) / (1 + exp(-2|x|))
    abs_x = tl.abs(x)
    mask_small = abs_x < 1.0
    mask_large = ~mask_small
    # Compute for small x
    x3 = x * x * x
    x5 = x3 * x * x
    x7 = x5 * x * x
    tanh_small = x - x3 / 3 + x5 / 5 - x7 / 7
    # Compute for large x
    exp_neg_2x = tl.exp(-2 * abs_x)
    tanh_large = (2 * exp_neg_2x) / (1 + exp_neg_2x)
    tanh_large = tl.where(mask_large, tl.sign(x) * (1 - tanh_large), 0.0)
    # Combine results
    tanh = tl.where(mask_small, tanh_small, tanh_large)
    # Store the result
    tl.store(out_ptr + offsets, tanh, mask=mask)


def triton_tanh(x: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.tanh(x)", call our Triton-based tanh
        return triton_tanh(x)