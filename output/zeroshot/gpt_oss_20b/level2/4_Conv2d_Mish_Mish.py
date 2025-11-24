import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mish_kernel(
    x_ptr,          # Pointer to input
    out_ptr,        # Pointer to output
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise Mish: x * tanh(log(1 + exp(x)))
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Softplus: log(1 + exp(x))
    sp = tl.log1p(tl.exp(x))

    # tanh(softplus)
    t = tl.tanh(sp)

    out = x * t

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that calls the Triton Mish kernel.
    """
    assert x.is_cuda, "Input must be on CUDA."
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by two Mish activations
    implemented with custom Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_mish(x)
        x = triton_mish(x)
        return x