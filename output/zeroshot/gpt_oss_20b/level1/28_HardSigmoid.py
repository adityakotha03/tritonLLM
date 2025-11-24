import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _hardsigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that applies the HardSigmoid function element‑wise:
        f(x) = min(max(x * 0.2 + 0.5, 0.0), 1.0)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load, compute and clamp
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.max(tl.min(x * 0.2 + 0.5, 1.0), 0.0)

    tl.store(out_ptr + offsets, y, mask=mask)

def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel for HardSigmoid.
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

class ModelNew(nn.Module):
    """
    Reimplementation of the original Model using a Triton kernel
    for the HardSigmoid activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)