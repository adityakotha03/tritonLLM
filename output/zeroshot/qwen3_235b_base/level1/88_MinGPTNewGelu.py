import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Constants
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
    coeff = 0.044715

    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_inner = tl.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + offsets, result, mask=mask)

def triton_gelu(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        return triton_gelu(x)

batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim).cuda()]

def get_init_inputs():
    return []