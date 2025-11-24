import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Leaky ReLU: x if x >= 0, negative_slope * x otherwise
    relu_out = tl.where(x >= 0, x, negative_slope * x)
    tl.store(out_ptr + offsets, relu_out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01):
    """
    Applies LeakyReLU activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimized block size for Ampere architecture (power of 2, balances memory and compute)

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    leaky_relu_kernel[grid](x, x, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_leaky_relu(x, negative_slope=self.negative_slope)