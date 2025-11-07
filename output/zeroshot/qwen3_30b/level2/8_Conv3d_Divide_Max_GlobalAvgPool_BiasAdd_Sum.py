import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr, BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    # Thread indices
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Output channel index
    pid_d = tl.program_id(2)  # Depth index
    pid_h = tl.program_id(3)  # Height index
    pid_w = tl.program_id(4)  # Width index

    # Block offsets
    off_b = pid_b * BLOCK_SIZE_B
    off_c = pid_c * BLOCK_SIZE_OC
    off_d = pid_d * BLOCK_SIZE_D
    off_h = pid_h * BLOCK_SIZE_H
    off_w = pid_w * BLOCK_SIZE_W

    # Load weights
    w_offset = off_c * kernel_d * kernel_h * kernel_w * in_channels + \
               tl.arange(0, BLOCK_SIZE_OC)[:, None, None, None] * kernel_d * kernel_h * kernel_w * in_channels + \
               tl.arange(0, kernel_d)[:, None, None] * kernel_h * kernel_w * in_channels + \
               tl.arange(0, kernel_h)[:, None] * kernel_w * in_channels + \
               tl.arange(0, kernel_w) * in_channels + \
               tl.arange(0, BLOCK_SIZE_IC)[None, None, None, :]  # [OC, KD, KH, KW, IC]

    w_ptrs = w_ptr + w_offset
    w_vals = tl.load(w_ptrs, mask=(tl.arange(0, BLOCK_SIZE_OC)[:, None, None, None] < out_channels) &
                                  (tl.arange(0, kernel_d)[:, None, None] < kernel_d) &
                                  (tl.arange(0, kernel_h)[:, None] < kernel_h) &
                                  (tl.arange(0, kernel_w) < kernel_w) &
                                  (tl.arange(0, BLOCK_SIZE_IC)[None, None, None, :] < in_channels), other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Compute input spatial offsets
    for di in range(0, kernel_d, 1):
        for hi in range(0, kernel_h, 1):
            for wi in range(0, kernel_w, 1):
                # Input spatial index
                i_d = off_d + di - pad_d
                i_h = off_h + hi - pad_h
                i_w = off_w + wi - pad_w

                # Load input
                i_offset = off_b * in_channels * depth * height * width + \
                           tl.arange(0, BLOCK_SIZE_IC)[None, None, None, :] * depth * height * width + \
                           i_d * height * width + \
                           i_h * width + \
                           i_w

                mask = (i_d >= 0) & (i_d < depth) & \
                       (i_h >= 0) & (i_h < height) & \
                       (i_w >= 0) & (i_w < width) & \
                       (tl.arange(0, BLOCK_SIZE_IC)[None, None, None, :] < in_channels)

                x_ptrs = x_ptr + i_offset
                x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

                # Accumulate
                acc += tl.dot(x_vals, w_vals[None, di, hi, wi, :])  # [IC] x [OC, IC] -> [OC]

    # Store output
    out_offset = off_b * out_channels * depth * height * width + \
                 off_c * depth * height * width + \
                 off_d * height * width + \
                 off_h * width + \
                 off_w

    out_ptrs = out_ptr + out_offset
    out_mask = (off_d < depth) & (off_h < height) & (off_w < width)
    out_mask = out_mask & (tl.arange(0, BLOCK_SIZE_OC)[:, None, None, None] < out_channels)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / divisor
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def max_pool3d_kernel(
    x_ptr,
    out_ptr,
    batch_size, in_channels, depth, height, width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    off_b = pid_b * BLOCK_SIZE_B
    off_c = pid_c * BLOCK_SIZE_C
    off_d = pid_d * BLOCK_SIZE_D
    off_h = pid_h * BLOCK_SIZE_H
    off_w = pid_w * BLOCK_SIZE_W

    # Output coordinates
    out_d = off_d // stride_d
    out_h = off_h // stride_h
    out_w = off_w // stride_w

    # Compute input range
    start_d = out_d * stride_d - pad_d
    start_h = out_h * stride_h - pad_h
    start_w = out_w * stride_w - pad_w
    end_d = start_d + kernel_d
    end_h = start_h + kernel_h
    end_w = start_w + kernel_w

    # Initialize output
    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over kernel
    for di in range(0, kernel_d, 1):
        for hi in range(0, kernel_h, 1):
            for wi in range(0, kernel_w, 1):
                i_d = start_d + di
                i_h = start_h + hi
                i_w = start_w + wi

                mask = (i_d >= 0) & (i_d < depth) & \
                       (i_h >= 0) & (i_h < height) & \
                       (i_w >= 0) & (i_w < width) & \
                       (tl.arange(0, BLOCK_SIZE_C)[None, None, None, :] < in_channels)

                x_offset = off_b * in_channels * depth * height * width + \
                           tl.arange(0, BLOCK_SIZE_C)[None, None, None, :] * depth * height * width + \
                           i_d * height * width + \
                           i_h * width + \
                           i_w

                x_vals = tl.load(x_ptr + x_offset, mask=mask, other=-float('inf'))
                acc = tl.maximum(acc, x_vals)

    # Store result
    out_offset = off_b * in_channels * (depth // stride_d) * (height // stride_h) * (width // stride_w) + \
                 off_c * (depth // stride_d) * (height // stride_h) * (width // stride_w) + \
                 off_d * (height // stride_h) * (width // stride_w) + \
                 off_h * (width // stride_w) + \
                 off_w

    out_mask = (off_d < depth // stride_d) & (off_h < height // stride_h) & (off_w < width // stride_w)
    out_mask = out_mask & (tl.arange(0, BLOCK_SIZE_C)[None, None, None, :] < in_channels)
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size, in_channels, depth, height, width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B
    off_c = pid_c * BLOCK_SIZE_C

    # Load all spatial elements
    spatial_size = depth * height * width
    x_offset = off_b * in_channels * spatial_size + \
               off_c * spatial_size + \
               tl.arange(0, spatial_size)

    x_vals = tl.load(x_ptr + x_offset, mask=(tl.arange(0, spatial_size) < spatial_size), other=0.0)
    avg_val = tl.sum(x_vals, axis=0) / spatial_size

    # Store output
    out_offset = off_b * in_channels + off_c
    out_ptrs = out_ptr + out_offset
    tl.store(out_ptrs, avg_val)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,
    batch_size, in_channels, depth, height, width,
    sum_dim,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B
    off_c = pid_c * BLOCK_SIZE_C

    # Determine spatial size
    if sum_dim == 1:
        spatial_size = depth * height * width
    else:
        spatial_size = 1

    # Flatten and sum
    x_offset = off_b * in_channels * spatial_size + \
               off_c * spatial_size + \
               tl.arange(0, spatial_size)

    x_vals = tl.load(x_ptr + x_offset, mask=(tl.arange(0, spatial_size) < spatial_size), other=0.0)
    sum_val = tl.sum(x_vals, axis=0)

    # Output
    out_offset = off_b * in_channels + off_c
    out_ptrs = out_ptr + out_offset
    tl.store(out_ptrs, sum_val)


def triton_conv3d(x, w, stride, padding):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = w.shape

    # Output shape
    out_depth = (depth + 2 * padding[0] - kernel_d) // stride[0] + 1
    out_height = (height + 2 * padding[1] - kernel_h) // stride[1] + 1
    out_width = (width + 2 * padding[2] - kernel_w) // stride[2] + 1

    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)

    # Define grid
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE_B"] - 1) // meta["BLOCK_SIZE_B"],
        (out_channels + meta["BLOCK_SIZE_OC"] - 1) // meta["BLOCK_SIZE_OC"],
        (out_depth + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (out_height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    conv3d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        BLOCK_SIZE_D=64, BLOCK_SIZE_H=64, BLOCK_SIZE_W=64,
        BLOCK_SIZE_OC=32, BLOCK_SIZE_IC=32,
        BLOCK_SIZE_B=1,
    )
    return out


def triton_div(x, divisor):
    out = torch.empty_like(x)
    grid = lambda meta: (x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    div_kernel[grid](x, out, x.numel(), divisor, BLOCK_SIZE=1024)
    return out


def triton_max_pool3d(x, kernel_size, stride, padding):
    batch_size, in_channels, depth, height, width = x.shape
    out_depth = (depth + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    out_height = (height + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    out_width = (width + 2 * padding[2] - kernel_size[2]) // stride[2] + 1

    out = torch.empty(batch_size, in_channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE_B"] - 1) // meta["BLOCK_SIZE_B"],
        (in_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        (out_depth + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (out_height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    max_pool3d_kernel[grid](
        x, out,
        batch_size, in_channels, depth, height, width,
        kernel_size[0], kernel_size[1], kernel_size[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        BLOCK_SIZE_D=16, BLOCK_SIZE_H=16, BLOCK_SIZE_W=16,
        BLOCK_SIZE_B=1, BLOCK_SIZE_C=16,
    )
    return out


def triton_global_avg_pool(x):
    batch_size, in_channels, depth, height, width = x.shape
    out = torch.empty(batch_size, in_channels, 1, 1, 1, dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE_B"] - 1) // meta["BLOCK_SIZE_B"],
        (in_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
    )

    global_avg_pool_kernel[grid](
        x, out,
        batch_size, in_channels, depth, height, width,
        BLOCK_SIZE_C=32, BLOCK_SIZE_B=1,
    )
    return out


def triton_add(x, y):
    out = torch.empty_like(x)
    grid = lambda meta: (x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
    return out


def triton_sum(x, dim):
    batch_size, in_channels, depth, height, width = x.shape
    if dim == 1:
        out = torch.empty(batch_size, in_channels, dtype=x.dtype, device=x.device)
        grid = lambda meta: (
            (batch_size + meta["BLOCK_SIZE_B"] - 1) // meta["BLOCK_SIZE_B"],
            (in_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        )
        sum_kernel[grid](x, out, batch_size, in_channels, depth, height, width, dim, BLOCK_SIZE_B=1, BLOCK_SIZE_C=32)
    else:
        out = torch.empty(batch_size, in_channels, 1, 1, 1, dtype=x.dtype, device=x.device)
        grid = lambda meta: (
            (batch_size + meta["BLOCK_SIZE_B"] - 1) // meta["BLOCK_SIZE_B"],
            (in_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        )
        sum_kernel[grid](x, out, batch_size, in_channels, depth, height, width, dim, BLOCK_SIZE_B=1, BLOCK_SIZE_C=32)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.divisor = divisor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # Step 1: Conv3d
        x = triton_conv3d(x, self.conv_weight, stride=(1, 1, 1), padding=(1, 1, 1))

        # Step 2: Division
        x = triton_div(x, self.divisor)

        # Step 3: Max Pool3d
        x = triton_max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        # Step 4: Global Average Pooling
        x = triton_global_avg_pool(x)

        # Step 5: Add bias
        x = triton_add(x, self.bias)

        # Step 6: Sum along dimension
        x = triton_sum(x, dim=self.sum_dim)

        return x