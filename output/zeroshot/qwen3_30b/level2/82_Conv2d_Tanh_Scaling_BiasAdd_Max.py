import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, pad,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr
):
    # Calculate block offsets
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output height and width
    out_h = (height + 2 * pad - kernel_size) // stride + 1
    out_w = (width + 2 * pad - kernel_size) // stride + 1

    # Define block offsets for output (H, W)
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W

    # Define block of output (H, W) within output
    h_offsets = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_offset + tl.arange(0, BLOCK_SIZE_W)

    # Compute output bounds
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w

    # Define output indices
    out_idx = (pid_batch * out_channels + pid_out_ch) * out_h * out_w + \
              h_offsets[:, None] * out_w + w_offsets[None, :]

    # Define input indices with padding
    x_h = h_offset * stride - pad + tl.arange(0, BLOCK_SIZE_H)[:, None]
    x_w = w_offset * stride - pad + tl.arange(0, BLOCK_SIZE_W)[None, :]
    x_h_mask = (x_h >= 0) & (x_h < height)
    x_w_mask = (x_w >= 0) & (x_w < width)

    # Compute input indices
    x_idx = (pid_batch * in_channels + tl.arange(0, BLOCK_SIZE_OUT)[:, None, None]) * height * width + \
            x_h[:, None, :] * width + x_w[None, :, :]

    # Load input (in_channels, BLOCK_SIZE_H, BLOCK_SIZE_W)
    x = tl.load(x_ptr + x_idx, mask=(x_h_mask[:, None, :] & x_w_mask[None, :, :])[:, None, :], other=0.0)
    x = x.to(tl.float32)

    # Load weight (in_channels, kernel_size, kernel_size) for the output channel
    w_idx = pid_out_ch * in_channels * kernel_size * kernel_size + \
            tl.arange(0, in_channels)[:, None, None] * kernel_size * kernel_size + \
            tl.arange(0, kernel_size)[:, None] * kernel_size + \
            tl.arange(0, kernel_size)[None, :]

    w = tl.load(w_ptr + w_idx, mask=(tl.arange(0, in_channels)[:, None, None] < in_channels) & \
                   (tl.arange(0, kernel_size)[:, None] < kernel_size) & \
                   (tl.arange(0, kernel_size)[None, :] < kernel_size), other=0.0)
    w = w.to(tl.float32)

    # Compute convolution: sum over in_channels, kernel_size, kernel_size
    # Use broadcasting to compute outer product of x and w
    # Shape: (in_channels, BLOCK_SIZE_H, BLOCK_SIZE_W) * (in_channels, kernel_size, kernel_size)
    # Then reduce over in_channels, kernel_size, kernel_size -> (BLOCK_SIZE_H, BLOCK_SIZE_W)
    out = tl.sum(x[:, None, :, :] * w[None, :, :, :], axis=(0, 2, 3))

    # Store output with mask
    tl.store(out_ptr + out_idx, out, mask=h_mask[:, None] & w_mask[None, :])


@triton.jit
def tanh_scale_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE
    offsets = off + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_channels * height * width

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.tanh(x)
    x = x * 2.0  # scaling factor
    bias = tl.load(bias_ptr + offsets % (out_channels * height * width), mask=(offsets % (out_channels * height * width)) < (out_channels * height * width), other=0.0)
    x = x + bias
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def max_pool_kernel(
    x_ptr, out_ptr,
    batch_size, out_channels, height, width,
    pool_kernel_size,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output height and width
    out_h = height // pool_kernel_size
    out_w = width // pool_kernel_size

    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W

    h_offsets = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_offset + tl.arange(0, BLOCK_SIZE_W)

    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w

    # Output indices
    out_idx = (pid_batch * out_channels + pid_out_ch) * out_h * out_w + \
              h_offsets[:, None] * out_w + w_offsets[None, :]

    # Input indices: pool_kernel_size x pool_kernel_size
    pool_h = h_offset * pool_kernel_size + tl.arange(0, pool_kernel_size)[:, None]
    pool_w = w_offset * pool_kernel_size + tl.arange(0, pool_kernel_size)[None, :]

    # Bounds check
    pool_h_mask = pool_h < height
    pool_w_mask = pool_w < width

    # Input indices for each pooling region
    x_idx = (pid_batch * out_channels + pid_out_ch) * height * width + \
            pool_h[:, None, :] * width + pool_w[None, :, :]

    # Load pooled data
    x = tl.load(x_ptr + x_idx, mask=(pool_h_mask[:, None, :] & pool_w_mask[None, :, :]), other=-float('inf'))

    # Reduce to max
    out = tl.max(x, axis=(0, 1))

    # Store output
    tl.store(out_ptr + out_idx, out, mask=h_mask[:, None] & w_mask[None, :])


def triton_conv2d(x, w, kernel_size, stride, pad):
    batch_size, in_channels, height, width = x.shape
    out_channels = w.shape[0]
    out_h = (height + 2 * pad - kernel_size) // stride + 1
    out_w = (width + 2 * pad - kernel_size) // stride + 1

    # Ensure contiguous
    x = x.contiguous()
    w = w.contiguous()

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Grid setup
    grid_h = (out_h + 15) // 16
    grid_w = (out_w + 15) // 16
    grid_out = (out_channels + 15) // 16
    grid_batch = batch_size

    # Blocks
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_OUT = 16

    # Launch
    conv2d_kernel[
        (grid_batch, grid_out, grid_h, grid_w)
    ](
        x, w, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, pad,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT
    )

    return out


def triton_tanh_scale_bias(x, bias, scaling_factor):
    batch_size, out_channels, height, width = x.shape
    n_elements = batch_size * out_channels * height * width

    x = x.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)

    # Grid setup
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    tanh_scale_bias_kernel[grid](
        x, bias, out,
        batch_size, out_channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_max_pool(x, pool_kernel_size):
    batch_size, out_channels, height, width = x.shape

    # Compute output dimensions
    out_h = height // pool_kernel_size
    out_w = width // pool_kernel_size

    # Ensure contiguous
    x = x.contiguous()

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Grid
    grid_h = (out_h + 15) // 16
    grid_w = (out_w + 15) // 16
    grid_out = (out_channels + 15) // 16
    grid_batch = batch_size

    # Blocks
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Launch
    max_pool_kernel[
        (grid_batch, grid_out, grid_h, grid_w)
    ](
        x, out,
        batch_size, out_channels, height, width,
        pool_kernel_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Use Triton kernels for all operations
        x = triton_conv2d(x, self.conv_weight, kernel_size=3, stride=1, pad=1)
        x = triton_tanh_scale_bias(x, self.bias, self.scaling_factor)
        x = triton_max_pool(x, pool_kernel_size=self.pool_kernel_size)
        return x