import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    input_stride3,
    weight_stride0,
    weight_stride1,
    weight_stride2,
    weight_stride3,
    output_stride0,
    output_stride1,
    output_stride2,
    output_stride3,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c_out = tl.program_id(3)

    # Compute output spatial indices
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_c_out = pid_c_out * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)

    # Broadcast output positions to all threads in block
    out_h = out_h[:, None, None]
    out_w = out_w[None, :, None]
    out_c_out = out_c_out[None, None, :]

    # Map to input coordinates (inverse of stride/padding)
    in_h = out_h * stride_h - padding_h
    in_w = out_w * stride_w - padding_w

    # Apply output padding if needed
    in_h += output_padding_h
    in_w += output_padding_w

    # Determine valid input regions
    mask_h = (in_h >= 0) & (in_h < input_height)
    mask_w = (in_w >= 0) & (in_w < input_width)
    mask_c_in = tl.arange(0, BLOCK_C_IN) < in_channels

    # Get batch indices
    batch_idx = pid_b * tl.arange(0, 1)  # Only one per block

    # Load weight: [out_c, in_c, kh, kw]
    w_ptrs = weight_ptr + (
        out_c_out[:, None, None] * weight_stride0 +
        tl.arange(0, BLOCK_C_IN)[None, :, None] * weight_stride1 +
        tl.arange(0, kernel_height)[None, None, :] * weight_stride2 +
        tl.arange(0, kernel_width)[None, None, :] * weight_stride3
    )

    w = tl.load(w_ptrs, mask=mask_c_in[None, :, None] & (tl.arange(0, kernel_height)[None, None, :] < kernel_height) & (tl.arange(0, kernel_width)[None, None, :] < kernel_width), other=0.0)

    # Load input data
    input_ptrs = input_ptr + (
        batch_idx[:, None, None, None] * input_stride0 +
        tl.arange(0, BLOCK_C_IN)[None, :, None, None] * input_stride1 +
        in_h[:, None, None, None] * input_stride2 +
        in_w[None, :, None, None] * input_stride3
    )

    x = tl.load(input_ptrs, mask=mask_h[:, None, None, None] & mask_w[None, :, None, None] & mask_c_in[None, :, None, None], other=0.0)

    # Compute convolution output: out_c_out × in_c × kh × kw → sum over in_c, kh, kw
    # Perform dot product between weights and input
    out = tl.sum(x * w, axis=(1, 2, 3))

    # Store output
    out_ptrs = output_ptr + (
        batch_idx[:, None, None, None] * output_stride0 +
        out_c_out[:, None, None, None] * output_stride1 +
        out_h[:, None, None, None] * output_stride2 +
        out_w[None, :, None, None] * output_stride3
    )
    tl.store(out_ptrs, out, mask=mask_h[:, None, None, None] & mask_w[None, :, None, None] & (out_c_out < out_channels))


@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    input_stride3,
    output_stride0,
    output_stride1,
    output_stride2,
    output_stride3,
    batch_size,
    channels,
    height,
    width,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Block offsets
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    # Check bounds
    h_mask = h < height
    w_mask = w < width
    c_mask = c < channels

    # Index into input
    input_ptrs = input_ptr + (
        pid_b * input_stride0 +
        c[:, None, None] * input_stride1 +
        h[:, None, None] * input_stride2 +
        w[None, :, None] * input_stride3
    )
    x = tl.load(input_ptrs, mask=h_mask[:, None, None] & w_mask[None, :, None] & c_mask[:, None, None], other=0.0)

    # Reduce over H and W
    mean = tl.sum(x, axis=(1, 2)) / (height * width)

    # Write output
    output_ptrs = output_ptr + (
        pid_b * output_stride0 +
        c[:, None, None] * output_stride1 +
        tl.arange(0, 1)[:, None, None] * output_stride2 +
        tl.arange(0, 1)[None, :, None] * output_stride3
    )
    tl.store(output_ptrs, mean, mask=c_mask[:, None, None])


@triton.jit
def multiply_scalar_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    input_stride3,
    output_stride0,
    output_stride1,
    output_stride2,
    output_stride3,
    batch_size,
    channels,
    height,
    width,
    multiplier,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    h_mask = h < height
    w_mask = w < width
    c_mask = c < channels

    input_ptrs = input_ptr + (
        pid_b * input_stride0 +
        c[:, None, None] * input_stride1 +
        h[:, None, None] * input_stride2 +
        w[None, :, None] * input_stride3
    )
    x = tl.load(input_ptrs, mask=h_mask[:, None, None] & w_mask[None, :, None] & c_mask[:, None, None], other=0.0)

    out = x * multiplier

    output_ptrs = output_ptr + (
        pid_b * output_stride0 +
        c[:, None, None] * output_stride1 +
        h[:, None, None] * output_stride2 +
        w[None, :, None] * output_stride3
    )
    tl.store(output_ptrs, out, mask=h_mask[:, None, None] & w_mask[None, :, None] & c_mask[:, None, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C_OUT': 32, 'BLOCK_C_IN': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C_OUT': 32, 'BLOCK_C_IN': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64, 'BLOCK_C_OUT': 16, 'BLOCK_C_IN': 16}, num_stages=4, num_warps=8),
    ],
    key=['in_channels', 'out_channels', 'input_height', 'input_width', 'kernel_height', 'kernel_width'],
)
def launch_conv_transpose_kernel(
    input, weight, output,
    input_stride0, input_stride1, input_stride2, input_stride3,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3,
    output_stride0, output_stride1, output_stride2, output_stride3,
    batch_size, in_channels, out_channels, input_height, input_width,
    kernel_height, kernel_width, stride_h, stride_w, padding_h, padding_w,
    output_padding_h, output_padding_w,
):
    grid = lambda meta: (
        batch_size,
        (input_height + meta['BLOCK_H'] - 1) // meta['BLOCK_H'],
        (input_width + meta['BLOCK_W'] - 1) // meta['BLOCK_W'],
        (out_channels + meta['BLOCK_C_OUT'] - 1) // meta['BLOCK_C_OUT']
    )
    conv_transpose_kernel[grid](
        input, weight, output,
        input_stride0, input_stride1, input_stride2, input_stride3,
        weight_stride0, weight_stride1, weight_stride2, weight_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_height, kernel_width, stride_h, stride_w, padding_h, padding_w,
        output_padding_h, output_padding_w,
        BLOCK_H=512, BLOCK_W=512, BLOCK_C_OUT=32, BLOCK_C_IN=32
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64, 'BLOCK_C': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_stages=4, num_warps=8),
    ],
    key=['height', 'width', 'channels'],
)
def launch_global_avg_pool_kernel(input, output, input_stride0, input_stride1, input_stride2, input_stride3,
                                 output_stride0, output_stride1, output_stride2, output_stride3,
                                 batch_size, channels, height, width, mode='double'):
    grid = lambda meta: (
        batch_size,
        (height + meta['BLOCK_H'] - 1) // meta['BLOCK_H'],
        (width + meta['BLOCK_W'] - 1) // meta['BLOCK_W'],
        (channels + meta['BLOCK_C'] - 1) // meta['BLOCK_C']
    )
    global_avg_pool_kernel[grid](
        input, output,
        input_stride0, input_stride1, input_stride2, input_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        batch_size, channels, height, width,
        BLOCK_H=64, BLOCK_W=64, BLOCK_C=64
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64, 'BLOCK_C': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 64}, num_stages=4, num_warps=8),
    ],
    key=['height', 'width', 'channels'],
)
def launch_multiply_scalar_kernel(input, output, input_stride0, input_stride1, input_stride2, input_stride3,
                                 output_stride0, output_stride1, output_stride2, output_stride3,
                                 batch_size, channels, height, width, multiplier):
    grid = lambda meta: (
        batch_size,
        (height + meta['BLOCK_H'] - 1) // meta['BLOCK_H'],
        (width + meta['BLOCK_W'] - 1) // meta['BLOCK_W'],
        (channels + meta['BLOCK_C'] - 1) // meta['BLOCK_C']
    )
    multiply_scalar_kernel[grid](
        input, output,
        input_stride0, input_stride1, input_stride2, input_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        batch_size, channels, height, width, multiplier,
        BLOCK_H=64, BLOCK_W=64, BLOCK_C=64
    )


def triton_conv_transpose(x, weight, stride, padding, output_padding):
    out_channels, in_channels, kh, kw = weight.shape
    batch_size, _, h, w = x.shape

    # Allocate output
    out = torch.empty(batch_size, out_channels, h * stride - 2 * padding + output_padding, w * stride - 2 * padding + output_padding, dtype=x.dtype, device=x.device)

    # Compute strides
    input_stride0, input_stride1, input_stride2, input_stride3 = x.stride()
    weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
    output_stride0, output_stride1, output_stride2, output_stride3 = out.stride()

    # Launch kernel
    launch_conv_transpose_kernel[
        (batch_size, (out.shape[2] + 63) // 64, (out.shape[3] + 63) // 64, (out_channels + 31) // 32)
    ](
        x, weight, out,
        input_stride0, input_stride1, input_stride2, input_stride3,
        weight_stride0, weight_stride1, weight_stride2, weight_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        batch_size, in_channels, out_channels, h, w,
        kh, kw, stride, stride, padding, padding,
        output_padding, output_padding,
    )

    return out


def triton_global_avg_pool(x, mode='double'):
    batch_size, channels, h, w = x.shape
    out = torch.empty(batch_size, channels, 1, 1, dtype=x.dtype, device=x.device)

    # Strides
    input_stride0, input_stride1, input_stride2, input_stride3 = x.stride()
    output_stride0, output_stride1, output_stride2, output_stride3 = out.stride()

    if mode == 'double':
        launch_global_avg_pool_kernel[
            (batch_size, (h + 63) // 64, (w + 63) // 64, (channels + 63) // 64)
        ](
            x, out,
            input_stride0, input_stride1, input_stride2, input_stride3,
            output_stride0, output_stride1, output_stride2, output_stride3,
            batch_size, channels, h, w
        )
    return out


def triton_multiply_scalar(x, multiplier):
    batch_size, channels, h, w = x.shape
    out = torch.empty_like(x)

    # Strides
    input_stride0, input_stride1, input_stride2, input_stride3 = x.stride()
    output_stride0, output_stride1, output_stride2, output_stride3 = out.stride()

    launch_multiply_scalar_kernel[
        (batch_size, (h + 63) // 64, (w + 63) // 64, (channels + 63) // 64)
    ](
        x, out,
        input_stride0, input_stride1, input_stride2, input_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        batch_size, channels, h, w, multiplier
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        # Use Triton kernel for conv_transpose
        x = triton_conv_transpose(x, self.conv_transpose.weight, self.stride, self.padding, self.output_padding)
        # Multiply scalar
        x = triton_multiply_scalar(x, self.multiplier)
        # First global average pooling
        x = triton_global_avg_pool(x, mode='double')
        # Second global average pooling
        x = triton_global_avg_pool(x, mode='double')
        return x