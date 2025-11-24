import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, name="B1024"),
        triton.Config({"BLOCK_SIZE": 2048}, name="B2048"),
        triton.Config({"BLOCK_SIZE": 4096}, name="B4096"),
    ],
    key=["n_elements"],
)
@triton.jit
def _softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise softplus: log(1 + exp(x)), with overflow protection.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Numerical stability: for large x, log(1+exp(x)) ~= x
    out = tl.where(x > 20.0, x, tl.log1p(tl.exp(x)))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Softplus activation to `x` using a Triton kernel.
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    assert x.dtype == torch.float32, "Only float32 dtype is supported."

    x = x.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)

    # Placeholder BLOCK_SIZE; actual value comes from autotune.
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that applies Softplus activation using a custom Triton kernel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)