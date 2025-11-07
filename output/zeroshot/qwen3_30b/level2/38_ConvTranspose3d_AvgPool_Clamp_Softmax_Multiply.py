import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr, y_ptr,
    batch_size, in_channels, depth, height, width,
    pool_depth, pool_height, pool_width,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr
):
    # Block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Block offsets
    d_offset = pid_d * BLOCK_SIZE_D
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W

    # Grid dimensions
    grid_d = (depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W

    # Calculate output indices
    out_d = d_offset + tl.arange(0, BLOCK_SIZE_D)
    out_h = h_offset + tl.arange(0, BLOCK_SIZE_H)
    out_w = w_offset + tl.arange(0, BLOCK_SIZE_W)

    # Masks for valid output indices
    mask_d = out_d < depth
    mask_h = out_h < height
    mask_w = out_w < width

    # Compute pool indices
    pool_d = out_d // pool_depth
    pool_h = out_h // pool_height
    pool_w = out_w // pool_width

    # Convert to global indices in input
    x_idx = (tl.broadcast_to(pool_d[:, None, None], (BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)) *
             stride_d + 
             tl.broadcast_to(pool_h[None, :, None], (BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)) *
             stride_h + 
             tl.broadcast_to(pool_w[None, None, :], (BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)) *
             stride_w)

    # Accumulate sum over pooled region
    # Use shared memory for intermediate accumulation
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for k in range(pool_depth * pool_height * pool_width):
        # Generate pooled input index
        idx = x_idx + k
        x_val = tl.load(x_ptr + idx, mask=(mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]), other=0.0)
        acc += x_val

    # Compute mean
    count = pool_depth * pool_height * pool_width
    out_val = acc / count

    # Write output
    out_idx = (out_d[:, None, None] * stride_d +
               out_h[None, :, None] * stride_h +
               out_w[None, None, :] * stride_w)
    tl.store(y_ptr + out_idx, out_val, mask=(mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]))


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_size, stride, padding,
    output_padding,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr
):
    # Thread block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Block offsets
    d_offset = pid_d * BLOCK_SIZE_D
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W

    # Output spatial indices
    out_d = d_offset + tl.arange(0, BLOCK_SIZE_D)
    out_h = h_offset + tl.arange(0, BLOCK_SIZE_H)
    out_w = w_offset + tl.arange(0, BLOCK_SIZE_W)

    # Masks
    mask_d = out_d < depth
    mask_h = out_h < height
    mask_w = out_w < width

    # Input spatial indices (for reverse kernel indexing)
    in_d = out_d * stride + padding
    in_h = out_h * stride + padding
    in_w = out_w * stride + padding

    # Block tiles for kernel (for shared memory)
    # Use shared memory for kernel weights
    tile_in_d = tl.arange(0, kernel_size)[:, None, None]
    tile_in_h = tl.arange(0, kernel_size)[None, :, None]
    tile_in_w = tl.arange(0, kernel_size)[None, None, :]

    # Input indices with offset from output
    in_idx_d = in_d[:, None, None] - tile_in_d
    in_idx_h = in_h[None, :, None] - tile_in_h
    in_idx_w = in_w[None, None, :] - tile_in_w

    # Masks for valid input indices
    mask_in_d = (in_idx_d >= 0) & (in_idx_d < depth)
    mask_in_h = (in_idx_h >= 0) & (in_idx_h < height)
    mask_in_w = (in_idx_w >= 0) & (in_idx_w < width)

    # Output tensor indices
    out_idx = (out_d[:, None, None] * stride_d +
               out_h[None, :, None] * stride_h +
               out_w[None, None, :] * stride_w)

    # Weight indices
    w_idx = tl.arange(0, in_channels)[:, None, None] * out_channels * kernel_size * kernel_size * kernel_size + \
            tl.arange(0, out_channels)[None, :, None] * kernel_size * kernel_size * kernel_size + \
            tile_in_d * kernel_size * kernel_size + \
            tile_in_h * kernel_size + \
            tile_in_w

    # Accumulate output
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for ic in range(in_channels):
        # Load input for current channel
        in_idx = (in_idx_d[:, :, :] * stride_d +
                  in_idx_h[:, :, :] * stride_h +
                  in_idx_w[:, :, :] * stride_w)
        x_val = tl.load(x_ptr + in_idx, mask=(mask_in_d & mask_in_h & mask_in_w), other=0.0)
        x_val = x_val[:, None, None, None]  # Add channel dim

        # Load weights
        w_val = tl.load(w_ptr + w_idx + ic * out_channels * kernel_size * kernel_size * kernel_size, mask=(mask_in_d & mask_in_h & mask_in_w), other=0.0)
        w_val = w_val[:, None, None, None]  # Add channel dim

        # Multiply and accumulate
        acc += x_val * w_val

    # Store output
    tl.store(out_ptr + out_idx[:, :, :, None] * out_channels + tl.arange(0, out_channels)[None, None, None, :], acc, mask=mask_d[:, None, None, None] & mask_h[None, :, None, None] & mask_w[None, None, :, None])


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    batch_size, channels, d, h, w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < d * h * w

    # Load row
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    # Exponentiate
    x_exp = tl.exp(x)
    # Compute sum
    x_sum = tl.sum(x_exp, axis=0)
    # Normalize
    out = x_exp / x_sum
    # Store
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def clamp_kernel(
    x_ptr, out_ptr,
    n_elements,
    min_val: tl.float32,
    max_val: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.clamp(x, min_val, max_val)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_avg_pool3d(x, kernel_size, stride=None, padding=0):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, in_channels, depth, height, width = x.shape
    stride = stride or kernel_size
    out_depth = (depth + 2 * padding - kernel_size) // stride + 1
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(batch_size, in_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    # Setup grid
    grid = lambda meta: (out_depth, out_height, out_width)

    # Use autotune for block size
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Run kernel
    avg_pool3d_kernel[grid](
        x, out,
        batch_size, in_channels, depth, height, width,
        kernel_size, kernel_size, kernel_size,
        BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W,
        stride, stride, stride
    )
    return out


def triton_conv_transpose3d(x, weight, stride, padding, output_padding):
    assert x.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    out_depth = (depth - 1) * stride + kernel_size - 2 * padding + output_padding
    out_height = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_width = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    # Grid
    grid = lambda meta: (out_depth, out_height, out_width)

    # Block size
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Run kernel
    conv_transpose3d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        output_padding,
        BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W,
        1, 1, 1
    )
    return out


def triton_softmax(x):
    batch_size, channels, d, h, w = x.shape
    x = x.view(batch_size * channels, -1)
    out = torch.empty_like(x)
    grid = lambda meta: (x.shape[1] + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
    BLOCK_SIZE = 128
    softmax_kernel[grid](x, out, batch_size * channels, channels, d, h, w, BLOCK_SIZE=BLOCK_SIZE)
    return out.view(batch_size, channels, d, h, w)


def triton_clamp(x, min_val, max_val):
    x = x.contiguous()
    out = torch.empty_like(x)
    grid = lambda meta: (x.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
    BLOCK_SIZE = 128
    clamp_kernel[grid](x, out, x.numel(), min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_mul(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    grid = lambda meta: (x.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
    BLOCK_SIZE = 128
    mul_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = triton_avg_pool3d(x, kernel_size=self.avg_pool.kernel_size, stride=self.avg_pool.stride, padding=self.avg_pool.padding)
        x = triton_conv_transpose3d(x, self.conv_transpose.weight, stride=self.conv_transpose.stride, padding=self.conv_transpose.padding, output_padding=self.conv_transpose.output_padding)
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = triton_softmax(x)
        x = x.view(b, c, d, h, w)
        x = triton_mul(x, self.scale)
        return x