import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,  # Strides for input
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,  # Strides for weight
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,  # Strides for output
    bias_stride_0, bias_stride_1, bias_stride_2,  # Strides for bias
    batch_size, out_channels, in_channels, height, width, out_height, out_width,
    kernel_size, stride, padding, output_padding,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute the output indices
    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W
    out_c_start = pid_c * BLOCK_SIZE_C

    # Output size
    h_mask = out_h_start + tl.arange(0, BLOCK_SIZE_H) < out_height
    w_mask = out_w_start + tl.arange(0, BLOCK_SIZE_W) < out_width
    c_mask = out_c_start + tl.arange(0, BLOCK_SIZE_C) < out_channels

    # Loop over input channels and kernel size for convolution
    accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    for k in range(0, in_channels):
        for kh in range(0, kernel_size):
            for kw in range(0, kernel_size):
                # Compute input position
                input_h = out_h_start - padding + kh * stride
                input_w = out_w_start - padding + kw * stride

                # Check bounds
                h_valid = (input_h >= 0) & (input_h < height)
                w_valid = (input_w >= 0) & (input_w < width)

                # Load input value if valid
                input_offset = (
                    pid_b * input_stride_0 +
                    k * input_stride_1 +
                    input_h * input_stride_2 +
                    input_w * input_stride_3
                )
                input_val = tl.load(
                    input_ptr + input_offset,
                    mask=h_valid & w_valid & h_mask[:, None] & w_mask[None, :],
                    other=0.0
                )

                # Load weight value
                weight_offset = (
                    out_c_start * weight_stride_0 +
                    k * weight_stride_1 +
                    kh * weight_stride_2 +
                    kw * weight_stride_3
                )
                weight_val = tl.load(
                    weight_ptr + weight_offset,
                    mask=c_mask[:, None, None] & h_mask[:, None, None] & w_mask[None, None, :]
                )

                # Accumulate
                accumulator += input_val[:, :, None] * weight_val[None, None, :]

    # Load bias
    bias_offset = pid_c * bias_stride_0
    bias_val = tl.load(bias_ptr + bias_offset, mask=c_mask, other=0.0)
    accumulator -= bias_val[None, None, :]

    # Apply tanh
    accumulator = tl.tanh(accumulator)

    # Store output
    output_offset = (
        pid_b * output_stride_0 +
        out_c_start * output_stride_1 +
        out_h_start * output_stride_2 +
        out_w_start * output_stride_3
    )
    tl.store(
        output_ptr + output_offset,
        accumulator,
        mask=h_mask[:, None, None] & w_mask[None, :] & c_mask[None, None, :]
    )


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride=2, padding=1, output_padding=1, kernel_size=4):
    # Ensure inputs are contiguous on GPU
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Get shapes
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Strides
    input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x.stride()
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3 = weight.stride()
    output_stride_0, output_stride_1, output_stride_2, output_stride_3 = out.stride()
    bias_stride_0, bias_stride_1, bias_stride_2 = bias.stride()

    # Tunable block sizes
    BLOCK_SIZE_H = 64
    BLOCK_SIZE_W = 64
    BLOCK_SIZE_C = 32

    # Grid setup
    grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid = (batch_size, grid_h, grid_w, grid_c)

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, bias, out,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        bias_stride_0, bias_stride_1, bias_stride_2,
        batch_size, out_channels, in_channels, height, width, out_height, out_width,
        kernel_size, stride, padding, output_padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Replace all operations with Triton kernel
        return triton_conv_transpose(x, self.conv_transpose.weight, self.bias, self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0], self.conv_transpose.kernel_size[0])