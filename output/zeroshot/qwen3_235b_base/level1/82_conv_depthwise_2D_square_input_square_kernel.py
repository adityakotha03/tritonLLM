import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,          # pointer to input tensor (NHWC layout)
    weight_ptr,     # pointer to weight tensor (K, K, C)
    bias_ptr,       # pointer to bias tensor (C,)
    output_ptr,     # pointer to output tensor (NHWC layout)
    batch_size,
    height,
    width,
    out_height,
    out_width,
    channels,
    kernel_size,
    stride,
    padding,
    has_bias: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute starting channel and number of channels per block
    c_offset = pid_c * BLOCK_C
    c_mask = c_offset + tl.arange(0, BLOCK_C) < channels
    c_range = tl.arange(0, BLOCK_C)

    # Compute output spatial block
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W
    h_range = h_offset + tl.arange(0, BLOCK_H)
    w_range = w_offset + tl.arange(0, BLOCK_W)

    # Load input values (only within valid bounds)
    input_h = h_range * stride - padding
    input_w = w_range * stride - padding
    ih_mask = (input_h[:, None] >= 0) & (input_h[:, None] < height) & \
              (input_w[None, :] >= 0) & (input_w[None, :] < width)
    ih = tl.where(ih_mask, input_h[:, None], 0)
    iw = tl.where(ih_mask, input_w[None, :], 0)

    # Base offset in input (NHWC)
    x_base = x_ptr + pid_batch * height * width * channels + ih[:, None] * width * channels + iw[None, :] * channels + c_offset
    x = tl.load(x_base + c_range[None, None, :], mask=ih_mask[:, :, None] & c_mask[None, None, :], other=0.0)

    # Initialize output
    output = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Convolution loop over kernel
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Compute input coordinates
            ih_k = input_h[:, None] + k_h
            iw_k = input_w[None, :] + k_w
            valid_h = (ih_k >= 0) & (ih_k < height)
            valid_w = (iw_k >= 0) & (iw_k < width)
            valid = valid_h & valid_w & ih_mask

            # Input index
            x_idx = x_ptr + pid_batch * height * width * channels + ih_k * width * channels + iw_k * channels + c_offset
            x_val = tl.load(x_idx + c_range[None, None, :], mask=valid[:, :, None] & c_mask[None, None, :], other=0.0)

            # Weight index
            w_idx = weight_ptr + k_h * kernel_size * channels + k_w * channels + c_offset
            w_val = tl.load(w_idx + c_range[None, None, :], mask=c_mask[None, None, :], other=0.0)

            # Multiply-accumulate
            output += x_val * w_val

    # Add bias if present
    if has_bias:
        bias = tl.load(bias_ptr + c_offset + c_range, mask=c_mask, other=0.0)
        output += bias[None, None, :]

    # Store output
    o_base = output_ptr + pid_batch * out_height * out_width * channels + h_range[:, None] * out_width * channels + w_range[None, :] * channels + c_offset
    tl.store(o_base + c_range[None, None, :], output, mask=(h_range[:, None] < out_height) & (w_range[None, :] < out_width) & c_mask[None, None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to NHWC for better memory coalescing
        x = x.to(memory_format=torch.channels_last)
        x = x.contiguous()
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        batch_size, height, width, channels = x_nhwc.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Output tensor
        output_nhwc = torch.empty((batch_size, out_height, out_width, channels), device=x.device, dtype=x.dtype)

        # Constants
        has_bias = self.bias is not None
        BLOCK_H = 16
        BLOCK_W = 16
        BLOCK_C = min(32, triton.cdiv(channels, 4))  # Multiple of 32 or smaller
        if channels % BLOCK_C != 0:
            BLOCK_C = 16  # Fallback

        # Grid
        grid = (batch_size, triton.cdiv(out_height, BLOCK_H), triton.cdiv(out_width, BLOCK_W), triton.cdiv(channels, BLOCK_C))

        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x_nhwc, self.weight, self.bias, output_nhwc,
            batch_size, height, width, out_height, out_width, channels,
            self.kernel_size, self.stride, self.padding,
            has_bias,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
        )

        # Convert back to NCHW
        output = output_nhwc.permute(0, 3, 1, 2).contiguous()
        return output