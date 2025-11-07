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
    # Each program instance handles a block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute x^3
    x3 = x * x * x

    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    coeff = tl.constexpr(0.044715)
    sqrt_2_over_pi = tl.constexpr(0.7978845608028654)  # sqrt(2/pi)
    x3_scaled = coeff * x3
    x_plus_x3 = x + x3_scaled
    intermediate = sqrt_2_over_pi * x_plus_x3
    tanh_val = tl.tanh(intermediate)
    one_plus_tanh = 1.0 + tanh_val
    out = 0.5 * x * one_plus_tanh

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    Triton-based GELU implementation with optimized block size and autotuning.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()
    
    # Define autotuning configuration
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        ],
        key=['n_elements'],
    )
    def kernel_launcher(grid, meta):
        gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=meta['BLOCK_SIZE'])

    # Determine grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    kernel_launcher[grid]()
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_gelu(x)