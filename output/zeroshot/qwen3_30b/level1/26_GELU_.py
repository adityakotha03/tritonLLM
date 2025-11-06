import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input with mask to avoid out-of-bounds access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Approximate using tanh-based approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # This is the standard approximation used in many implementations

    # Constants
    sqrt_2_over_pi = tl.constexpr(0.7978845608028654)  # sqrt(2/pi)
    coeff = tl.constexpr(0.044715)

    # Compute x^3
    x3 = x * x * x
    # Compute x + coeff * x^3
    x_plus_cubed = x + coeff * x3
    # Compute sqrt(2/pi) * (x + coeff * x^3)
    scaled = sqrt_2_over_pi * x_plus_cubed
    # Compute tanh of that
    tanh_val = tl.tanh(scaled)
    # Compute 0.5 * x * (1 + tanh_val)
    result = 0.5 * x * (1.0 + tanh_val)

    # Store the output
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
    ],
    key=['n_elements'],
)
def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Use dynamic grid based on input size and BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel with autotuned BLOCK_SIZE
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=triton.runtime.autotuner.get_config().BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)