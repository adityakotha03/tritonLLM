import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mish_tanh_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # We fuse Mish + Tanh into a single pass
    # Step 1: Compute softplus(x) = ln(1 + exp(x))
    # Use tl.where to avoid exp overflow
    exp_x = tl.exp(x)
    softplus = tl.where(x >= 0, x + tl.log(1 + tl.exp(-x)), tl.log(1 + exp_x))

    # Step 2: Compute tanh(softplus(x)) for Mish
    tanh_softplus = tl.tanh(softplus)
    mish = x * tanh_softplus

    # Step 3: Apply tanh on top: tanh(mish)
    result = tl.tanh(mish)

    tl.store(out_ptr + offsets, result, mask=mask)


def triton_mish_tanh(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mish_tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses Conv3d with Mish and Tanh using Triton kernel for activation fusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = triton_mish_tanh(x)
        return x