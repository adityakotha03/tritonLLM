import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh(softplus(x)) = tanh(ln(1 + exp(x)))
    # softplus(x) = ln(1 + exp(x))
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    tanh_sp = tl.tanh(softplus)
    result = x * tanh_sp

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def hardtanh_scale_add_kernel(
    x_ptr, out_ptr, add_value, scale, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x + add_value
    x = tl.where(x < -1.0, -1.0, x)
    x = tl.where(x > 1.0, 1.0, x)
    x = x * scale

    tl.store(out_ptr + offsets, x, mask=mask)


def triton_mish(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_hardtanh_add_scale(x: torch.Tensor, add_value: float, scale: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    hardtanh_scale_add_kernel[grid](x, out, add_value, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Mish activation and fused Hardtanh+Add+Scale.
    The transposed convolution remains as-is since it's highly optimized in PyTorch/CuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_mish(x)
        x = triton_hardtanh_add_scale(x, self.add_value, self.scale)
        return x