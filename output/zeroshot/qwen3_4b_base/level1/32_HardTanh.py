import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply HardTanh: clamp to [-1, 1]
    # HardTanh(x) = x if -1 <= x <= 1, else -1 or 1
    # We compute this directly using conditional logic
    # Note: Triton supports conditional operations via `tl.where`
    lower_mask = x < -1.0
    upper_mask = x > 1.0
    # Clamp values
    clamped = tl.where(lower_mask, -1.0, tl.where(upper_mask, 1.0, x))
    # Store result
    tl.store(out_ptr + offsets, clamped, mask=mask)


def triton_hardtanh(x: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel implementation of HardTanh activation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimal block size for Ampere, power of 2, balances memory and compute

    # Grid size: number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    hardtanh_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim)

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return triton_hardtanh(x)