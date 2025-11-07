import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Thread block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Compute output spatial coordinates
    d = pid_d * TILE_SIZE + tl.arange(0, TILE_SIZE)[:, None]
    h = pid_h * TILE_SIZE + tl.arange(0, TILE_SIZE)[None, :]
    w = pid_w * TILE_SIZE + tl.arange(0, TILE_SIZE)[None, None, :]

    # Compute output indices
    d_out = d * stride_depth - padding_depth
    h_out = h * stride_height - padding_height
    w_out = w * stride_width - padding_width

    # Compute input indices
    k_d = tl.arange(0, kernel_depth)[:, None, None]
    k_h = tl.arange(0, kernel_height)[None, :, None]
    k_w = tl.arange(0, kernel_width)[None, None, :]

    # Compute total offsets for input
    d_in = d_out[None, :, :] + k_d
    h_in = h_out[None, :, :] + k_h
    w_in = w_out[None, :, :] + k_w

    # Create masks for valid input indices
    d_mask = (d_in >= 0) & (d_in < depth)
    h_mask = (h_in >= 0) & (h_in < height)
    w_mask = (w_in >= 0) & (w_in < width)
    valid_mask = d_mask & h_mask & w_mask

    # Get batch and channel indices
    pid_b = tl.program_id(3)
    pid_c = tl.program_id(4)

    # Compute input and output offsets
    x_offset = (pid_b * in_channels + tl.arange(0, in_channels))[:, None, None, None] * depth * height * width
    w_offset = (pid_c * in_channels + tl.arange(0, in_channels))[:, None, None, None] * kernel_depth * kernel_height * kernel_width
    out_offset = (pid_b * out_channels + pid_c) * (depth + 2 * padding_depth) * (height + 2 * padding_height) * (width + 2 * padding_width)

    # Load input and weight
    x_vals = tl.load(
        x_ptr + x_offset + d_in * height * width + h_in * width + w_in,
        mask=valid_mask[:, None, :, :],
        other=0.0
    ).to(tl.float32)

    w_vals = tl.load(
        w_ptr + w_offset + k_d * kernel_height * kernel_width + k_h * kernel_width + k_w,
        mask=valid_mask[:, None, :, :],
        other=0.0
    ).to(tl.float32)

    # Perform convolution
    out_vals = tl.sum(x_vals * w_vals, axis=0)

    # Store output
    out_offsets = out_offset + d * height * width + h * width + w
    tl.store(out_ptr + out_offsets, out_vals, mask=(d < (depth + 2 * padding_depth)) & (h < (height + 2 * padding_height)) & (w < (width + 2 * padding_width)))


@triton.jit
def max_pool_3d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)
    pid_b = tl.program_id(4)

    d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    d = tl.clip(d, 0, depth - 1)
    h = tl.clip(h, 0, height - 1)
    w = tl.clip(w, 0, width - 1)

    # Kernel sizes
    k_d = tl.arange(0, kernel_size_d)[:, None, None]
    k_h = tl.arange(0, kernel_size_h)[None, :, None]
    k_w = tl.arange(0, kernel_size_w)[None, None, :]

    # Input indices
    d_in = d[None, :, :] + k_d
    h_in = h[None, :, :] + k_h
    w_in = w[None, :, :] + k_w

    d_in = tl.clip(d_in, 0, depth - 1)
    h_in = tl.clip(h_in, 0, height - 1)
    w_in = tl.clip(w_in, 0, width - 1)

    # Compute input offset
    input_offsets = (pid_b * channels + pid_c) * depth * height * width + d_in * height * width + h_in * width + w_in
    input_vals = tl.load(x_ptr + input_offsets, mask=(d_in < depth) & (h_in < height) & (w_in < width), other=-float('inf'))

    # Compute max
    out_vals = tl.max(input_vals, axis=0)

    # Output offset
    output_offsets = (pid_b * channels + pid_c) * (depth // kernel_size_d) * (height // kernel_size_h) * (width // kernel_size_w) + (d // kernel_size_d) * (height // kernel_size_h) * (width // kernel_size_w) + (h // kernel_size_h) * (width // kernel_size_w) + (w // kernel_size_w)
    tl.store(out_ptr + output_offsets, out_vals)


@triton.jit
def sum_dim_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    target_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Assume target_dim is 1 (sum over channel)
    d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Clamp to valid indices
    d = tl.clip(d, 0, depth - 1)
    h = tl.clip(h, 0, height - 1)
    w = tl.clip(w, 0, width - 1)

    # Compute input offsets
    input_offsets = (pid_b * channels + tl.arange(0, channels))[:, None, None, None] * depth * height * width + d * height * width + h * width + w
    input_vals = tl.load(x_ptr + input_offsets, mask=(d < depth) & (h < height) & (w < width), other=0.0)

    # Sum over channel
    out_val = tl.sum(input_vals, axis=0)

    # Output offset
    output_offsets = (pid_b * 1 + 0) * (depth // 1) * (height // 1) * (width // 1) + d * height * width + h * width + w
    tl.store(out_ptr + output_offsets, out_val)


def triton_conv_transpose_3d(x, w, out_shape, stride, padding, kernel_size):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = w.shape

    # Output spatial dimensions
    out_depth = (depth - 1) * stride[0] + kernel_depth - 2 * padding[0]
    out_height = (height - 1) * stride[1] + kernel_height - 2 * padding[1]
    out_width = (width - 1) * stride[2] + kernel_width - 2 * padding[2]

    # Allocate output
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)

    # Grid
    grid_d = (out_depth + 31) // 32
    grid_h = (out_height + 31) // 32
    grid_w = (out_width + 31) // 32
    grid_c = (out_channels + 31) // 32
    grid_b = (batch_size + 31) // 32

    # Launch kernel
    conv_transpose_3d_kernel[
        (grid_d, grid_h, grid_w, grid_b, grid_c),
        BLOCK_SIZE=32,
        TILE_SIZE=32
    ](
        x, w, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_depth, kernel_height, kernel_width,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
    )
    return out


def triton_max_pool_3d(x, kernel_size, stride):
    batch_size, channels, depth, height, width = x.shape
    out_depth = (depth - kernel_size[0]) // stride[0] + 1
    out_height = (height - kernel_size[1]) // stride[1] + 1
    out_width = (width - kernel_size[2]) // stride[2] + 1

    out = torch.empty(batch_size, channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)

    grid_d = (out_depth + 31) // 32
    grid_h = (out_height + 31) // 32
    grid_w = (out_width + 31) // 32
    grid_c = (channels + 31) // 32
    grid_b = (batch_size + 31) // 32

    max_pool_3d_kernel[
        (grid_d, grid_h, grid_w, grid_c, grid_b),
        BLOCK_SIZE=32
    ](
        x, out, batch_size, channels,
        depth, height, width,
        kernel_size[0], kernel_size[1], kernel_size[2],
    )
    return out


def triton_sum_dim(x, dim, keepdim=True):
    batch_size, channels, depth, height, width = x.shape
    if dim == 1:
        out = torch.empty(batch_size, 1, depth, height, width, dtype=x.dtype, device=x.device)
        grid_d = (depth + 31) // 32
        grid_h = (height + 31) // 32
        grid_w = (width + 31) // 32
        grid_b = (batch_size + 31) // 32
        grid_c = (channels + 31) // 32

        sum_dim_kernel[
            (grid_b, grid_c, grid_d, grid_h, grid_w),
            BLOCK_SIZE=32
        ](
            x, out, batch_size, channels, depth, height, width, 1
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        # Custom Triton kernels
        x = triton_conv_transpose_3d(
            x, self.conv_transpose.weight, 
            (x.shape[0], self.conv_transpose.out_channels, 
             (x.shape[2] - 1) * self.stride[0] + self.kernel_size[0] - 2 * self.padding[0],
             (x.shape[3] - 1) * self.stride[1] + self.kernel_size[1] - 2 * self.padding[1],
             (x.shape[4] - 1) * self.stride[2] + self.kernel_size[2] - 2 * self.padding[2]),
            self.stride, self.padding, self.kernel_size
        )
        x = triton_max_pool_3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        x = triton_max_pool_3d(x, kernel_size=(3, 3, 3), stride=(3, 3, 3))
        x = triton_sum_dim(x, dim=1, keepdim=True)
        return x