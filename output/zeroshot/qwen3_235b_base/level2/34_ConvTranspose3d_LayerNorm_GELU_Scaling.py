import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    batch_stride, n_channels, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * batch_stride + tl.arange(0, BLOCK_SIZE)
    mask = offset < (pid * batch_stride + n_channels)

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_channels
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_channels
    inv_var = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    output = (x - mean) * inv_var * weight + bias

    tl.store(y_ptr + offset, output, mask=mask)


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh
    gelu = x * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, gelu, mask=mask)


@triton.jit
def mul_kernel(x_ptr, out_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size = x.shape[0]
    n_channels = x.shape[1]
    spatial_size = x.shape[2] * x.shape[3] * x.shape[4]
    total_size = n_channels * spatial_size
    batch_stride = total_size

    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_channels)

    def grid(meta):
        return (batch_size * spatial_size,)

    layer_norm_kernel[grid](
        x, weight, bias, out,
        batch_stride, n_channels, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_mul_scalar(x: torch.Tensor, scalar: float):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mul_kernel[grid](x, out, scalar, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for layer norm, GELU, and scalar multiplication.
    ConvTranspose3d remains as-is due to complexity and limited benefit from custom Triton implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

        # Register weight and bias for Triton kernel access
        self.register_buffer("ln_weight", self.layer_norm.weight)
        self.register_buffer("ln_bias", self.layer_norm.bias)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # Move channel to last dimension for layer norm
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x = triton_layer_norm(x, self.ln_weight, self.ln_bias, self.layer_norm.eps)
        x = x.view(x_shape).permute(0, 4, 1, 2, 3).contiguous()  # Restore original layout
        x = triton_gelu(x)
        x = triton_mul_scalar(x, self.scaling_factor)
        return x