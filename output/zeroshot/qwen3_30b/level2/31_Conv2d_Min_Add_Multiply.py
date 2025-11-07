import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_min_add_scale_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weights pointer
    bias_ptr,  # Bias pointer
    out_ptr,  # Output tensor pointer
    x_stride0, x_stride1, x_stride2, x_stride3,  # Strides for input (B, C, H, W)
    w_stride0, w_stride1, w_stride2, w_stride3,  # Strides for weights (O, I, K, K)
    bias_stride0, bias_stride1, bias_stride2,  # Strides for bias (O, 1, 1)
    out_stride0, out_stride1, out_stride2, out_stride3,  # Strides for output (B, O, H, W)
    batch_size, in_channels, out_channels, height, width, kernel_size, constant_value, scaling_factor,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output tile dimensions
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    c_start = pid_c * BLOCK_C

    # Compute output offsets
    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    offs_c = c_start + tl.arange(0, BLOCK_C)

    # Mask to prevent out-of-bounds access
    h_mask = offs_h < height
    w_mask = offs_w < width
    c_mask = offs_c < out_channels

    # Initialize output tile
    out = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Load bias (broadcast across spatial dims)
    bias = tl.load(bias_ptr + offs_c[:, None, None] * bias_stride0,
                   mask=offs_c[:, None, None] < out_channels,
                   other=0.0)

    # Iterate over input channels and kernel
    for ci in range(0, in_channels, 1):  # in_channels is not large enough to require tiling
        # Load input channel tile (aligned with spatial dims)
        x_block = tl.load(
            x_ptr + (pid_b * x_stride0 + ci * x_stride1 + offs_h[:, None, None] * x_stride2 + offs_w[None, :, None] * x_stride3),
            mask=(h_mask[:, None, None] & w_mask[None, :, None]),
            other=0.0
        )

        # Load kernel tile (H x W)
        w_block = tl.load(
            w_ptr + (pid_c * w_stride0 + ci * w_stride1 + offs_h[:, None, None] * w_stride2 + offs_w[None, :, None] * w_stride3),
            mask=(h_mask[:, None, None] & w_mask[None, :, None]),
            other=0.0
        )

        # Perform convolution: sum over spatial dimensions
        out += tl.sum(x_block * w_block, axis=(0, 1))  # [H, W] -> scalar per output channel

    # Add bias and apply min with constant
    out = out + bias
    out = tl.minimum(out, constant_value)

    # Multiply by scaling factor
    out = out * scaling_factor

    # Store result
    tl.store(
        out_ptr + (pid_b * out_stride0 + pid_c * out_stride1 + offs_h[:, None, None] * out_stride2 + offs_w[None, :, None] * out_stride3),
        out,
        mask=(h_mask[:, None, None] & w_mask[None, :, None] & c_mask[:, None, None])
    )


def triton_conv_min_add_scale(x, weight, bias, constant_value, scaling_factor):
    # Ensure contiguous tensors on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Input dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3 = weight.stride()
    bias_stride0, bias_stride1, bias_stride2 = bias.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = (batch_size, out_channels, height, width)

    # Output tensor
    out = torch.empty_like(x, size=(batch_size, out_channels, height, width))

    # Define block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 16

    # Grid dimensions
    grid = (
        batch_size,
        (out_channels + BLOCK_C - 1) // BLOCK_C,
        (height + BLOCK_H - 1) // BLOCK_H,
        (width + BLOCK_W - 1) // BLOCK_W
    )

    # Launch kernel
    conv_min_add_scale_kernel[grid](
        x, weight, bias, out,
        x_stride0, x_stride1, x_stride2, x_stride3,
        w_stride0, w_stride1, w_stride2, w_stride3,
        bias_stride0, bias_stride1, bias_stride2,
        out_stride0, out_stride1, out_stride2, out_stride3,
        batch_size, in_channels, out_channels, height, width, kernel_size, constant_value, scaling_factor,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)  # No bias in conv, we handle it in kernel
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Replace the standard conv + min + add + mul with a single Triton kernel
        return triton_conv_min_add_scale(x, self.conv.weight, self.bias, self.constant_value, self.scaling_factor)