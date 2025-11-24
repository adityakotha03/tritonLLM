import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_channels, spatial_size,
    stride_batch_c, stride_c_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)

    offset_c = tl.arange(0, BLOCK_SIZE)
    mask = offset_c < n_channels

    input_offsets = batch_idx * stride_batch_c + spatial_idx + offset_c * stride_c_spatial
    x = tl.load(input_ptr + input_offsets, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum

    tl.store(output_ptr + input_offsets, x_softmax, mask=mask)


@triton.jit
def subtract_swish_kernel(
    input_ptr, subtract_ptr, output_ptr,
    n_channels, spatial_size,
    stride_batch_c, stride_c_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)

    offset_c = tl.arange(0, BLOCK_SIZE)
    mask = offset_c < n_channels

    input_offsets = batch_idx * stride_batch_c + spatial_idx + offset_c * stride_c_spatial
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    sub = tl.load(subtract_ptr + offset_c, mask=mask, other=0.0)
    x_sub = x - sub
    sigmoid = tl.sigmoid(x_sub)
    swish = x_sub * sigmoid

    tl.store(output_ptr + input_offsets, swish, mask=mask)


@triton.jit
def max_reduce_channel_kernel(
    input_ptr, output_ptr,
    n_channels, spatial_size,
    stride_batch_c, stride_c_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)

    offset_c = tl.arange(0, BLOCK_SIZE)
    mask = offset_c < n_channels

    input_offsets = batch_idx * stride_batch_c + spatial_idx + offset_c * stride_c_spatial
    x = tl.load(input_ptr + input_offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)

    output_offset = batch_idx * spatial_size + spatial_idx
    tl.store(output_ptr + output_offset, x_max)


def triton_softmax(x: torch.Tensor, dim: int):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_channels = x.shape[dim]
    spatial_size = x.numel() // x.shape[0] // n_channels
    stride_batch_c = x.stride(0)
    stride_c_spatial = x.stride(dim)

    def grid(meta):
        return (x.shape[0], spatial_size)

    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    softmax_kernel[grid](
        x, out,
        n_channels, spatial_size,
        stride_batch_c, stride_c_spatial,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_subtract_swish(x: torch.Tensor, subtract: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_channels = x.shape[1]
    spatial_size = x.numel() // x.shape[0] // n_channels
    stride_batch_c = x.stride(0)
    stride_c_spatial = x.stride(1)

    def grid(meta):
        return (x.shape[0], spatial_size)

    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    subtract_swish_kernel[grid](
        x, subtract, out,
        n_channels, spatial_size,
        stride_batch_c, stride_c_spatial,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_max_reduce_channel(x: torch.Tensor, dim: int):
    x = x.contiguous()
    n_channels = x.shape[dim]
    output_shape = list(x.shape)
    output_shape.pop(dim)
    out = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    spatial_size = x.numel() // x.shape[0] // n_channels
    stride_batch_c = x.stride(0)
    stride_c_spatial = x.stride(dim)

    def grid(meta):
        return (x.shape[0], spatial_size)

    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    max_reduce_channel_kernel[grid](
        x, out,
        n_channels, spatial_size,
        stride_batch_c, stride_c_spatial,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for:
        - Softmax
        - Subtract + Swish (fused)
        - Max reduction over channels
    ConvTranspose3d and MaxPool3d are kept as native PyTorch ops due to complexity and availability of optimized CuDNN kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = triton_softmax(x, dim=1)
        x = triton_subtract_swish(x, self.subtract)
        x = triton_max_reduce_channel(x, dim=1)
        return x