import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for HardTanh activation
@triton.jit
def hardtanh_kernel(
    x_ptr: tl.constexpr,
    out_ptr: tl.constexpr,
    n_elements: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Starting offset for this program (block)
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Compute offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for boundaries
    mask = offsets < n_elements

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Clamp to [-1, 1]
    x = tl.where(x < min_val, min_val, x)
    x = tl.where(x > max_val, max_val, x)

    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_hardtanh(x: torch.Tensor) -> torch.Tensor:
    """
    Apply HardTanh activation using the custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    out = torch.empty_like(x)

    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch Triton kernel
    hardtanh_kernel[grid](
        x,
        out,
        n_elements,
        -1.0,  # min_val
        1.0,   # max_val
        BLOCK_SIZE=128,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that applies HardTanh activation using a Triton kernel.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x)