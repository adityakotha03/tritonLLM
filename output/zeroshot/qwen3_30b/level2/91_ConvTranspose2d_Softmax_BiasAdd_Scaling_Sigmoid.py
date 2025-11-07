import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding, output_padding,
    input_stride0, input_stride1, input_stride2, input_stride3,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3,
    output_stride0, output_stride1, output_stride2, output_stride3,
    bias_stride0, bias_stride1, bias_stride2,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr, BLOCK_OC: tl.constexpr
):
    # Grid configuration: each block processes a tile of output
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_ic = tl.program_id(2)
    pid_oc = tl.program_id(3)

    # Compute output tile bounds
    start_h = pid_h * BLOCK_H
    start_w = pid_w * BLOCK_W
    start_ic = pid_ic * BLOCK_IC
    start_oc = pid_oc * BLOCK_OC

    # Output size after transposed conv
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Block indices for output
    offs_h = start_h + tl.arange(0, BLOCK_H)
    offs_w = start_w + tl.arange(0, BLOCK_W)
    offs_ic = start_ic + tl.arange(0, BLOCK_IC)
    offs_oc = start_oc + tl.arange(0, BLOCK_OC)

    # Mask for out-of-bounds
    mask_h = offs_h < out_h
    mask_w = offs_w < out_w
    mask_ic = offs_ic < in_channels
    mask_oc = offs_oc < out_channels

    # Load bias
    bias = tl.load(bias_ptr + offs_oc[:, None, None] * bias_stride0, mask=mask_oc[:, None, None], other=0.0)

    # Accumulator for output
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_OC), dtype=tl.float32)

    # Loop over input channels
    for k in range(0, in_channels, BLOCK_IC):
        k_offset = k
        k_mask = k_offset + tl.arange(0, BLOCK_IC) < in_channels

        # Load input tile
        input_tile = tl.load(
            input_ptr + (
                (k_offset + offs_ic[:, None, None]) * input_stride1 +
                offs_h[:, None, None] * input_stride2 +
                offs_w[:, None, None] * input_stride3
            ),
            mask=(mask_ic[:, None, None] & mask_h[:, None, None] & mask_w[:, None, None]),
            other=0.0
        )

        # Load weight tile
        weight_tile = tl.load(
            weight_ptr + (
                (start_oc + offs_oc[:, None, None, None]) * weight_stride0 +
                (k_offset + offs_ic[None, :, None, None]) * weight_stride1 +
                (tl.arange(0, kernel_size)[:, None, None, None]) * weight_stride2 +
                (tl.arange(0, kernel_size)[None, None, :, None]) * weight_stride3
            ),
            mask=(mask_oc[:, None, None, None] & k_mask[None, :, None, None]),
            other=0.0
        )

        # Conv transpose via im2col + matmul
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute output indices
                oh = offs_h[:, None, None] * stride + kh - padding
                ow = offs_w[:, None, None] * stride + kw - padding
                o_mask_h = (oh >= 0) & (oh < height)
                o_mask_w = (ow >= 0) & (ow < width)

                # Apply mask
                oh = oh * o_mask_h + (0 if padding == 0 else (height - 1)) * (~o_mask_h)
                ow = ow * o_mask_w + (0 if padding == 0 else (width - 1)) * (~o_mask_w)

                # Clamp to valid bounds
                oh = tl.clip(oh, 0, height - 1)
                ow = tl.clip(ow, 0, width - 1)

                # Load input at (oh, ow)
                input_val = tl.load(
                    input_ptr + (
                        (k_offset + offs_ic[:, None, None]) * input_stride1 +
                        oh * input_stride2 +
                        ow * input_stride3
                    ),
                    mask=(mask_ic[:, None, None] & o_mask_h & o_mask_w),
                    other=0.0
                )

                # Weight contribution
                weight_val = weight_tile[:, :, kh, kw]
                acc += input_val[:, :, None] * weight_val[None, None, :]

    # Add bias
    acc += bias[None, None, :]

    # Store output
    tl.store(
        output_ptr + (
            (start_h + offs_h[:, None, None]) * output_stride2 +
            (start_w + offs_w[None, :, None]) * output_stride3 +
            (start_oc + offs_oc[None, None, :]) * output_stride1
        ),
        acc,
        mask=(mask_h[:, None, None] & mask_w[None, :, None] & mask_oc[None, None, :])
    )


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    start_h = pid_h * BLOCK_H
    start_w = pid_w * BLOCK_W

    offs_h = start_h + tl.arange(0, BLOCK_H)
    offs_w = start_w + tl.arange(0, BLOCK_W)

    mask_h = offs_h < height
    mask_w = offs_w < width

    for c in range(0, channels, 32):
        # Load data
        offs_c = c + tl.arange(0, 32)
        mask_c = offs_c < channels

        # Load values
        x = tl.load(
            x_ptr + (
                offs_c[:, None, None] * x_stride1 +
                offs_h[:, None, None] * x_stride2 +
                offs_w[None, :, None] * x_stride3
            ),
            mask=(mask_c[:, None, None] & mask_h[:, None, None] & mask_w[None, :, None]),
            other=-float('inf')
        )

        # Compute max
        x_max = tl.max(x, axis=0)
        x_max = tl.max(x_max, axis=1)  # Max across spatial dims
        x_max = tl.broadcast_to(x_max[:, None, None], (32, BLOCK_H, BLOCK_W))
        x = x - x_max

        # Compute exp
        x_exp = tl.exp(x)

        # Compute sum
        x_sum = tl.sum(x_exp, axis=0)
        x_sum = tl.sum(x_sum, axis=1)  # Sum across spatial dims
        x_sum = tl.broadcast_to(x_sum[None, None, :], (32, BLOCK_H, BLOCK_W))
        x_exp = x_exp / x_sum

        # Store
        tl.store(
            out_ptr + (
                offs_c[:, None, None] * out_stride1 +
                offs_h[:, None, None] * out_stride2 +
                offs_w[None, :, None] * out_stride3
            ),
            x_exp,
            mask=(mask_c[:, None, None] & mask_h[:, None, None] & mask_w[None, :, None])
        )


@triton.jit
def scale_and_sigmoid_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    scaling_factor: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    start_h = pid_h * BLOCK_H
    start_w = pid_w * BLOCK_W

    offs_h = start_h + tl.arange(0, BLOCK_H)
    offs_w = start_w + tl.arange(0, BLOCK_W)

    mask_h = offs_h < height
    mask_w = offs_w < width

    # Load and scale
    x = tl.load(
        x_ptr + (
            offs_h[:, None, None] * x_stride2 +
            offs_w[None, :, None] * x_stride3
        ),
        mask=(mask_h[:, None, None] & mask_w[None, :, None]),
        other=0.0
    )
    x = x * scaling_factor

    # Apply sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    # Store
    tl.store(
        out_ptr + (
            offs_h[:, None, None] * out_stride2 +
            offs_w[None, :, None] * out_stride3
        ),
        x,
        mask=(mask_h[:, None, None] & mask_w[None, :, None])
    )


def triton_conv_transpose(x, weight, bias, kernel_size, stride, padding, output_padding):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kh, kw = weight.shape

    out_h = (height - 1) * stride + kh - 2 * padding + output_padding
    out_w = (width - 1) * stride + kw - 2 * padding + output_padding

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Define grid
    BLOCK_H, BLOCK_W = 16, 16
    BLOCK_IC, BLOCK_OC = 32, 32

    grid_h = (out_h + BLOCK_H - 1) // BLOCK_H
    grid_w = (out_w + BLOCK_W - 1) // BLOCK_W
    grid_ic = (in_channels + BLOCK_IC - 1) // BLOCK_IC
    grid_oc = (out_channels + BLOCK_OC - 1) // BLOCK_OC

    # Launch kernel
    conv_transpose_kernel[
        (grid_h, grid_w, grid_ic, grid_oc)
    ](
        x, weight, out, bias,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, output_padding,
        *x.stride(), *weight.stride(), *out.stride(), *bias.stride(),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_IC=BLOCK_IC, BLOCK_OC=BLOCK_OC
    )

    return out


def triton_softmax(x):
    batch_size, channels, height, width = x.shape
    x = x.contiguous()

    out = torch.empty_like(x)

    BLOCK_H, BLOCK_W = 16, 16

    grid_h = (height + BLOCK_H - 1) // BLOCK_H
    grid_w = (width + BLOCK_W - 1) // BLOCK_W

    softmax_kernel[
        (grid_h, grid_w)
    ](
        x, out,
        batch_size, channels, height, width,
        *x.stride(), *out.stride(),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )

    return out


def triton_scale_and_sigmoid(x, scaling_factor):
    batch_size, channels, height, width = x.shape
    x = x.contiguous()

    out = torch.empty_like(x)

    BLOCK_H, BLOCK_W = 16, 16

    grid_h = (height + BLOCK_H - 1) // BLOCK_H
    grid_w = (width + BLOCK_W - 1) // BLOCK_W

    scale_and_sigmoid_kernel[
        (grid_h, grid_w)
    ](
        x, out,
        batch_size, channels, height, width,
        *x.stride(), *out.stride(),
        scaling_factor=scaling_factor,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Replace all operations with Triton kernels
        x = triton_conv_transpose(x, self.conv_transpose.weight, self.bias, self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0])
        x = triton_softmax(x)
        x = triton_scale_and_sigmoid(x, self.scaling_factor)
        return x