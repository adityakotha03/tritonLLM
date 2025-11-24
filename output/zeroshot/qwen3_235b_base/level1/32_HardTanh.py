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
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.minimum(max_val, tl.maximum(min_val, x))
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    hardtanh_kernel[grid](
        x,
        out,
        n_elements,
        min_val=min_val,
        max_val=max_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x, min_val=-1.0, max_val=1.0)