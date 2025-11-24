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

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute GELU: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Precompute constants
    sqrt_2_over_pi = tl.math.sqrt(2.0 / tl.math.pi)
    coeff = 0.044715

    # Compute x^3
    x3 = x * x * x
    # Compute x + 0.044715 * x^3
    inner = x + coeff * x3
    # Compute tanh(sqrt(2/pi) * inner)
    tanh_inner = tl.math.tanh(sqrt_2_over_pi * inner)
    # Final GELU output
    out = 0.5 * x * (1.0 + tanh_inner)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    Custom GELU implementation using Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimized for Ampere architecture: power of 2, good for tensor cores and memory coalescing

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Replace the original GELU with custom Triton kernel
        return triton_gelu(x)

batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return []