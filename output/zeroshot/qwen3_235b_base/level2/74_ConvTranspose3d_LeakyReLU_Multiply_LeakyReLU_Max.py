import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr, out_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x >= 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.2):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    BLOCK_SIZE = 1024
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def mul_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mul(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    BLOCK_SIZE = 1024
    mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def max_pool3d_kernel(
    x_ptr,
    out_ptr,
    n_batches,
    channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    stride_d,
    stride_h,
    stride_w,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_hw = tl.program_id(3)

    # Compute output spatial indices
    out_h = out_hw // output_width
    out_w = out_hw % output_width

    # Input indices
    in_d_start = out_d * stride_d
    in_h_start = out_h * stride_h
    in_w_start = out_w * stride_w

    # Define block sizes
    d_offsets = in_d_start + tl.arange(0, BLOCK_SIZE_D)
    h_offsets = in_h_start + tl.arange(0, BLOCK_SIZE_HW)
    w_offsets = in_w_start + tl.arange(0, BLOCK_SIZE_HW)

    # Mask valid indices
    d_mask = d_offsets < input_depth
    h_mask = h_offsets < input_height
    w_mask = w_offsets < input_width

    # Broadcast masks
    d_mask = tl.reshape(d_mask, (BLOCK_SIZE_D, 1, 1))
    h_mask = tl.reshape(h_mask, (1, BLOCK_SIZE_HW, 1))
    w_mask = tl.ones((BLOCK_SIZE_D, BLOCK_SIZE_HW, BLOCK_SIZE_HW), dtype=tl.int1)  # Full block for w
    mask = d_mask and h_mask and w_mask

    # Compute linear indices
    input_idx = (
        batch_idx * channels * input_depth * input_height * input_width +
        channel_idx * input_depth * input_height * input_width +
        d_offsets[:, None, None] * input_height * input_width +
        h_offsets[None, :, None] * input_width +
        w_offsets[None, None, :]
    )
    input_idx = tl.reshape(input_idx, (BLOCK_SIZE_D * BLOCK_SIZE_HW * BLOCK_SIZE_HW,))

    # Load data
    data = tl.load(x_ptr + input_idx, mask=tl.reshape(mask, (BLOCK_SIZE_D * BLOCK_SIZE_HW * BLOCK_SIZE_HW,)), other=-float('inf'))

    # Max reduce
    pool_max = tl.max(data)

    # Output index
    output_idx = (
        batch_idx * channels * output_depth * output_height * output_width +
        channel_idx * output_depth * output_height * output_width +
        out_d * output_height * output_width +
        out_h * output_width +
        out_w
    )
    tl.store(out_ptr + output_idx, pool_max)


def triton_max_pool3d(x: torch.Tensor, kernel_size=2, stride=2):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    b, c, d, h, w = x.shape
    out_d = (d - kernel_size) // stride + 1
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    out = torch.empty((b, c, out_d, out_h, out_w), dtype=x.dtype, device=x.device)

    grid = (b, c, out_d, out_h * out_w)
    max_pool3d_kernel[grid](
        x,
        out,
        b, c, d, h, w,
        out_d, out_h, out_w,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        BLOCK_SIZE_D=4,
        BLOCK_SIZE_HW=8,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for LeakyReLU, elementwise multiply, and MaxPool3d.
    ConvTranspose3d is kept as PyTorch (uses cuDNN), but pointwise ops are fused via Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.2

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_leaky_relu(x, self.negative_slope)
        x = triton_mul(x, self.multiplier)
        x = triton_leaky_relu(x, self.negative_slope)
        x = triton_max_pool3d(x, kernel_size=2, stride=2)
        return x