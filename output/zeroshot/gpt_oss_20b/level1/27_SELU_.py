import torch
import torch.nn as nn
import triton
import triton.language as tl

# SELU constants (float32)
SCALE = tl.constexpr(1.0507009873554804934193349852946)
ALPHA = tl.constexpr(1.6732632423543772848170429916717)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def selu_kernel(
    x_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements (coalesced)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # SELU computation
    pos = x > 0.0
    out = tl.where(pos, SCALE * x, SCALE * ALPHA * (tl.exp(x) - 1.0))

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_selu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)

    # Default block size (autotune will override)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)