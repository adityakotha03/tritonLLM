import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def elu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ELU: x if x >= 0 else alpha * (exp(x) - 1)
    is_negative = x < 0
    pos = x
    neg = alpha * (tl.exp(x) - 1.0)
    out = tl.where(is_negative, neg, pos)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_elu(x: torch.Tensor, alpha: float = 1.0):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Choose block size - 1024 is efficient for large tensors on A100
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    elu_kernel[grid](x, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized version of ELU model using custom Triton kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        return triton_elu(x, self.alpha)