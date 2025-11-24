import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def div_leaky_relu_kernel(
    x_ptr, out_ptr, divisor, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_div = x / divisor
    out = tl.where(x_div >= 0, x_div, x_div * 0.01)

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_div_leaky_relu(x: torch.Tensor, divisor: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    div_leaky_relu_kernel[grid](x, out, divisor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with fused division and LeakyReLU using Triton.
    The convolution remains as standard PyTorch (uses cuDNN), but the elementwise
    operations are fused into a single Triton kernel to reduce memory bandwidth usage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        x = triton_div_leaky_relu(x, self.divisor)
        return x