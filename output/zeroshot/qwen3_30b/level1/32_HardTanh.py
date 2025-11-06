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
    min_val: tl.float32,
    max_val: tl.float32,
):
    # Compute the block offset
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply HardTanh: clamp values between min_val and max_val
    out = tl.clamp(x, min_val, max_val)

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Applies HardTanh activation using a custom Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        min_val (float): Minimum value for clamping.
        max_val (float): Maximum value for clamping.

    Returns:
        torch.Tensor: Output tensor with HardTanh applied, same shape as input.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements
    n_elements = x.numel()
    # Use a BLOCK_SIZE that is a power of two and optimized for A100
    BLOCK_SIZE = 1024  # Good balance for A100 shared memory and occupancy

    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    hardtanh_kernel[grid](
        x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, min_val=min_val, max_val=max_val
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x, min_val=-1.0, max_val=1.0)