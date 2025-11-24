import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
    is_positive = x > 0.0
    negative_branch = alpha * (tl.exp(x) - 1.0)
    out = tl.where(is_positive, x, negative_branch)
    out = scale * out
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_selu(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Autotune BLOCK_SIZE
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)