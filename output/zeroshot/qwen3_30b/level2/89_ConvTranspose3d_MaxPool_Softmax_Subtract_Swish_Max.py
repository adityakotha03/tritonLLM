import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, weight_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_size, stride, padding, output_padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block index
    pid = tl.program_id(0)
    # Number of total output elements
    total_elements = batch_size * out_channels * (depth + output_padding) * (height + output_padding) * (width + output_padding)
    # Each block handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute output indices
    o_batch = offsets // (out_channels * (depth + output_padding) * (height + output_padding) * (width + output_padding))
    o_c = (offsets % (out_channels * (depth + output_padding) * (height + output_padding) * (width + output_padding))) // ((depth + output_padding) * (height + output_padding) * (width + output_padding))
    o_d = (offsets % ((depth + output_padding) * (height + output_padding) * (width + output_padding))) // ((height + output_padding) * (width + output_padding))
    o_h = (offsets % ((height + output_padding) * (width + output_padding))) // (width + output_padding)
    o_w = offsets % (width + output_padding)

    # Map output indices to input indices
    i_d = o_d * stride - padding
    i_h = o_h * stride - padding
    i_w = o_w * stride - padding

    # Loop over input channels and kernel
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(kernel_size):
        for j in range(kernel_size):
            for i in range(kernel_size):
                # Compute input indices with bounds checking
                i_d_idx = i_d + i
                i_h_idx = i_h + j
                i_w_idx = i_w + k
                # Check if valid input indices
                valid = (i_d_idx >= 0) & (i_d_idx < depth) & \
                        (i_h_idx >= 0) & (i_h_idx < height) & \
                        (i_w_idx >= 0) & (i_w_idx < width)

                # Load input and weight
                if valid:
                    # Compute offsets for input and weight
                    x_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
                               (i_d_idx * height * width + i_h_idx * width + i_w_idx)
                    w_offset = (o_c * in_channels * kernel_size * kernel_size * kernel_size + \
                                (i * in_channels + o_c) * kernel_size * kernel_size + \
                                j * kernel_size + k)
                    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                    w_val = tl.load(weight_ptr + w_offset, mask=mask, other=0.0)
                    acc += x_val * w_val

    # Store output
    out_offset = (o_batch * out_channels + o_c) * (depth + output_padding) * (height + output_padding) * (width + output_padding) + \
                 (o_d * (height + output_padding) * (width + output_padding) + o_h * (width + output_padding) + o_w)
    tl.store(out_ptr + out_offset, acc, mask=mask)


@triton.jit
def max_pool_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels,
    depth, height, width,
    pool_kernel_size, pool_stride, pool_padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * in_channels * depth * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute output indices
    o_batch = offsets // (in_channels * depth * height * width)
    o_c = (offsets % (in_channels * depth * height * width)) // (depth * height * width)
    o_d = (offsets % (depth * height * width)) // (height * width)
    o_h = (offsets % (height * width)) // width
    o_w = offsets % width

    # Compute input indices
    i_d = o_d * pool_stride - pool_padding
    i_h = o_h * pool_stride - pool_padding
    i_w = o_w * pool_stride - pool_padding

    # Compute pool area bounds
    i_d_end = i_d + pool_kernel_size
    i_h_end = i_h + pool_kernel_size
    i_w_end = i_w + pool_kernel_size

    # Initialize accumulator
    acc = tl.float32(0)
    for di in range(pool_kernel_size):
        for dj in range(pool_kernel_size):
            for dk in range(pool_kernel_size):
                # Check bounds
                valid = (i_d + di >= 0) & (i_d + di < depth) & \
                        (i_h + dj >= 0) & (i_h + dj < height) & \
                        (i_w + dk >= 0) & (i_w + dk < width)

                if valid:
                    # Compute input index
                    i_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
                               ((i_d + di) * height * width + (i_h + dj) * width + (i_w + dk))
                    val = tl.load(x_ptr + i_offset, mask=mask, other=tl.float32(-float('inf')))
                    acc = tl.max(acc, val)

    # Store output
    out_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
                 (o_d * height * width + o_h * width + o_w)
    tl.store(out_ptr + out_offset, acc, mask=mask)


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels,
    depth, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * in_channels * depth * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute channel indices
    o_batch = offsets // (in_channels * depth * height * width)
    o_c = (offsets % (in_channels * depth * height * width)) // (depth * height * width)
    o_d = (offsets % (depth * height * width)) // (height * width)
    o_h = (offsets % (height * width)) // width
    o_w = offsets % width

    # Get channel-specific max
    channel_max = tl.float32(0)
    for c in range(in_channels):
        c_offset = (o_batch * in_channels + c) * (depth * height * width) + \
                   (o_d * height * width + o_h * width + o_w)
        val = tl.load(x_ptr + c_offset, mask=mask, other=tl.float32(-float('inf')))
        channel_max = tl.max(channel_max, val)

    # Compute exponential
    acc = tl.float32(0)
    for c in range(in_channels):
        c_offset = (o_batch * in_channels + c) * (depth * height * width) + \
                   (o_d * height * width + o_h * width + o_w)
        val = tl.load(x_ptr + c_offset, mask=mask, other=tl.float32(0))
        exp_val = tl.exp(val - channel_max)
        acc += exp_val

    # Normalize
    for c in range(in_channels):
        c_offset = (o_batch * in_channels + c) * (depth * height * width) + \
                   (o_d * height * width + o_h * width + o_w)
        val = tl.load(x_ptr + c_offset, mask=mask, other=tl.float32(0))
        out_val = tl.exp(val - channel_max) / acc
        tl.store(out_ptr + c_offset, out_val, mask=mask)


@triton.jit
def subtract_kernel(
    x_ptr, sub_ptr, out_ptr,
    batch_size, in_channels,
    depth, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * in_channels * depth * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute channel indices
    o_batch = offsets // (in_channels * depth * height * width)
    o_c = (offsets % (in_channels * depth * height * width)) // (depth * height * width)
    o_d = (offsets % (depth * height * width)) // (height * width)
    o_h = (offsets % (height * width)) // width
    o_w = offsets % width

    # Compute input offset
    x_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
               (o_d * height * width + o_h * width + o_w)
    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    # Compute subtraction offset
    sub_offset = o_c
    sub_val = tl.load(sub_ptr + sub_offset, mask=mask, other=0.0)

    # Store result
    out_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
                 (o_d * height * width + o_h * width + o_w)
    tl.store(out_ptr + out_offset, x_val - sub_val, mask=mask)


@triton.jit
def swish_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels,
    depth, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * in_channels * depth * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute indices
    o_batch = offsets // (in_channels * depth * height * width)
    o_c = (offsets % (in_channels * depth * height * width)) // (depth * height * width)
    o_d = (offsets % (depth * height * width)) // (height * width)
    o_h = (offsets % (height * width)) // width
    o_w = offsets % width

    # Load input
    x_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
               (o_d * height * width + o_h * width + o_w)
    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    # Apply swish: x * sigmoid(x)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-x_val))
    out_val = x_val * sigmoid_val

    # Store output
    out_offset = (o_batch * in_channels + o_c) * (depth * height * width) + \
                 (o_d * height * width + o_h * width + o_w)
    tl.store(out_ptr + out_offset, out_val, mask=mask)


@triton.jit
def max_channel_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels,
    depth, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * depth * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute indices
    o_batch = offsets // (depth * height * width)
    o_d = (offsets % (depth * height * width)) // (height * width)
    o_h = (offsets % (height * width)) // width
    o_w = offsets % width

    # Initialize accumulator
    acc = tl.float32(-float('inf'))
    for c in range(in_channels):
        c_offset = (o_batch * in_channels + c) * (depth * height * width) + \
                   (o_d * height * width + o_h * width + o_w)
        val = tl.load(x_ptr + c_offset, mask=mask, other=tl.float32(-float('inf')))
        acc = tl.max(acc, val)

    # Store result
    out_offset = (o_batch * depth * height * width) + (o_d * height * width + o_h * width + o_w)
    tl.store(out_ptr + out_offset, acc, mask=mask)


def triton_conv_transpose(x, weight, bias=None):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    stride = 2
    padding = 1
    output_padding = 1

    # Output shape
    out_d = (depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_h = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w = (width - 1) * stride - 2 * padding + kernel_size + output_padding

    # Allocate output
    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    # Grid setup
    total_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out


def triton_max_pool(x, kernel_size, stride, padding):
    batch_size, in_channels, depth, height, width = x.shape
    out_d = (depth + 2 * padding - kernel_size) // stride + 1
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(batch_size, in_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    total_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_pool_kernel[grid](
        x, out,
        batch_size, in_channels,
        depth, height, width,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_softmax(x):
    batch_size, in_channels, depth, height, width = x.shape
    out = torch.empty_like(x)

    total_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softmax_kernel[grid](
        x, out,
        batch_size, in_channels,
        depth, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_subtract(x, sub):
    batch_size, in_channels, depth, height, width = x.shape
    out = torch.empty_like(x)

    total_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    subtract_kernel[grid](
        x, sub, out,
        batch_size, in_channels,
        depth, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_swish(x):
    batch_size, in_channels, depth, height, width = x.shape
    out = torch.empty_like(x)

    total_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    swish_kernel[grid](
        x, out,
        batch_size, in_channels,
        depth, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_max_channel(x):
    batch_size, in_channels, depth, height, width = x.shape
    out = torch.empty(batch_size, depth, height, width, device=x.device, dtype=x.dtype)

    total_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_channel_kernel[grid](
        x, out,
        batch_size, in_channels,
        depth, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        # Initialize conv transpose weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if out_channels > 0 else None
        self.subtract = nn.Parameter(torch.randn(out_channels))

        # Store hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        # Ensure inputs are on GPU and contiguous
        x = x.contiguous()

        # Apply ConvTranspose3d via Triton
        x = triton_conv_transpose(x, self.weight, self.bias)

        # Apply MaxPool3d via Triton
        x = triton_max_pool(x, self.pool_kernel_size, self.pool_stride, self.pool_padding)

        # Apply Softmax across channels via Triton
        x = triton_softmax(x)

        # Subtract bias via Triton
        x = triton_subtract(x, self.subtract)

        # Apply Swish via Triton
        x = triton_swish(x)

        # Max across channels via Triton
        x = triton_max_channel(x)

        return x