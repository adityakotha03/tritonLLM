import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU approximation using tanh method: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x_cubed)  # sqrt(2/pi) ~ 0.7978845608028654
    tanh_inner = tl.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_inner)
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_gelu(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def clamp_add_leaky_relu_kernel(
    x_ptr, sum_tensor_ptr, out_ptr, n_elements, sum_tensor_size, negative_slope, min_val, max_val,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Broadcast sum_tensor: sum_tensor has shape (out_channels, 1, 1, 1)
    # We map linear index -> channel index
    channel_idx = (offsets // (sum_tensor_size[1] * sum_tensor_size[2] * sum_tensor_size[3])) % sum_tensor_size[0]
    sum_val = tl.load(sum_tensor_ptr + channel_idx, mask=mask, other=0.0)
    x = x + sum_val
    # LeakyReLU: x if x >= 0 else x * negative_slope
    x_relu = tl.where(x >= 0, x, x * negative_slope)
    # Clamp
    x_clamped = tl.clamp(x_relu, min_val, max_val)
    tl.store(out_ptr + offsets, x_clamped, mask=mask)


def triton_clamp_add_leaky_relu(x, sum_tensor, negative_slope=0.2, min_val=-1.0, max_val=1.0):
    assert x.is_cuda and sum_tensor.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    sum_tensor = sum_tensor.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    clamp_add_leaky_relu_kernel[grid](
        x, sum_tensor, out, n_elements, sum_tensor.shape, negative_slope, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for fused clamp, add, leaky_relu and GELU operations.
    The 3D convolution remains as-is since it's highly optimized in PyTorch (uses cuDNN),
    but the subsequent pointwise operations are fused into custom Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = triton_clamp_add_leaky_relu(x, self.sum_tensor, negative_slope=0.2, min_val=-1.0, max_val=1.0)
        x = triton_gelu(x)
        return x