import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # HardSigmoid: x / 6 + 0.5, clamped to [0, 1]
    # We compute: x / 6 + 0.5, then clamp to [0, 1]
    x_div_6 = x / 6.0
    out = x_div_6 + 0.5
    # Clamp to [0, 1]
    out = tl.where(out < 0.0, 0.0, out)
    out = tl.where(out > 1.0, 1.0, out)

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardsigmoid(x: torch.Tensor):
    """
    Custom Triton kernel implementation of HardSigmoid activation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimized for Ampere architecture, power of 2

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty_like(x)
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)