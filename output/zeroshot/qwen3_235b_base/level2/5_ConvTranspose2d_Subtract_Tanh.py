import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    bias_batch_stride, bias_channel_stride,
    in_batch_stride, in_channel_stride, in_height_stride, in_width_stride,
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
    weight_height_stride, weight_width_stride,
    input_height, input_width, output_height, output_width,
    in_channels, out_channels, kernel_size,
    stride, padding, output_padding,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (tl.cdiv(out_channels, BLOCK_SIZE_M))
    c_out_idx = pid % (tl.cdiv(out_channels, BLOCK_SIZE_M))

    offset_m = c_out_idx * BLOCK_SIZE_M
    offset_n = batch_idx * BLOCK_SIZE_N

    # Pointers for output tiles
    out_tile_ptr = out_ptr + offset_m * out_channel_stride + offset_n * out_batch_stride

    # Load bias into shared memory
    bias_ptrs = bias_ptr + (offset_m + tl.arange(0, BLOCK_SIZE_M)) * bias_channel_stride
    bias_mask = (offset_m + tl.arange(0, BLOCK_SIZE_M)) < out_channels
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    for h in range(0, output_height):
        for w in range(0, output_width):
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for c_in_group in range(0, in_channels // groups):
                for ky in range(0, kernel_size):
                    for kx in range(0, kernel_size):
                        h_in = h * stride - padding + ky
                        w_in = w * stride - padding + kx

                        # Bounds check
                        h_in_valid = (h_in >= 0) & (h_in < input_height)
                        w_in_valid = (w_in >= 0) & (w_in < input_width)

                        # Compute input and weight offsets
                        x_ptrs = (
                            x_ptr +
                            (offset_n + tl.arange(0, BLOCK_SIZE_N)) * in_batch_stride +
                            c_in_group * in_channel_stride +
                            h_in * in_height_stride +
                            w_in * in_width_stride
                        )
                        mask_x = h_in_valid & w_in_valid & ((offset_n + tl.arange(0, BLOCK_SIZE_N)) < batch_size)
                        x = tl.load(x_ptrs, mask=mask_x, other=0.0)

                        weight_ptrs = (
                            weight_ptr +
                            (offset_m + tl.arange(0, BLOCK_SIZE_M)) * out_channel_stride +
                            c_in_group * weight_height_stride * weight_width_stride +
                            ky * weight_width_stride +
                            kx
                        )
                        weight = tl.load(weight_ptrs, mask=bias_mask, other=0.0)

                        # Outer product
                        acc += weight[:, None] * x[None, :]

            # Store output tile
            out_offsets = h * out_height_stride + w * out_width_stride
            out_ptrs = out_tile_ptr + out_offsets
            out_mask = ((offset_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < out_channels) & \
                       ((offset_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < batch_size)
            out = acc.to(tl.float16)
            tl.store(out_ptrs, out, mask=out_mask)

            # Add bias and apply tanh
            out_with_bias = tl.tanh(out + bias[:, None])
            tl.store(out_ptrs, out_with_bias, mask=out_mask)


def triton_conv_transpose_tanh(x, weight, bias, stride, padding, output_padding):
    batch_size, _, input_height, input_width = x.shape
    out_channels, in_channels, kernel_size, _ = weight.shape
    output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding

    out = torch.empty((batch_size, out_channels, output_height, output_width), device=x.device, dtype=torch.float16)
    x = x.to(torch.float16)
    weight = weight.to(torch.float16)
    bias = bias.to(torch.float16)

    def grid(META):
        return (triton.cdiv(out_channels, META['BLOCK_SIZE_M']) * batch_size,)

    _conv_transpose_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr(),
        out_ptr=out.data_ptr(),
        bias_batch_stride=0,
        bias_channel_stride=bias.stride(0),
        in_batch_stride=x.stride(0),
        in_channel_stride=x.stride(1),
        in_height_stride=x.stride(2),
        in_width_stride=x.stride(3),
        out_batch_stride=out.stride(0),
        out_channel_stride=out.stride(1),
        out_height_stride=out.stride(2),
        out_width_stride=out.stride(3),
        weight_height_stride=weight.stride(2),
        weight_width_stride=weight.stride(3),
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=1,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for fused transposed convolution, bias subtraction, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return triton_conv_transpose_tanh(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding
        )