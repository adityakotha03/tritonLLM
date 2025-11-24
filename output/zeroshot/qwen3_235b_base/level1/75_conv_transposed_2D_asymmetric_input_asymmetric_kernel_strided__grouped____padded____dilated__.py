import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, out_height, out_width,
    in_channels_per_group, input_height, input_width,
    kernel_height, kernel_width,
    stride_height, stride_width,
    padding_height, padding_width,
    dilation_height, dilation_width,
    groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)
    pid_g = tl.program_id(3)

    # Calculate output spatial dimensions
    oh = pid_oh
    ow = pid_ow

    # Pointers to output
    output_offset_base = pid_b * out_channels * out_height * out_width + oh * out_width * out_channels + ow * out_channels
    out_ptrs = out_ptr + output_offset_base + tl.arange(0, BLOCK_SIZE_N)[:, None] * out_channels + tl.arange(0, BLOCK_SIZE_M)[None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over input channels per group
    for ic in range(0, in_channels_per_group, BLOCK_SIZE_K):
        # Pointers to input
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Compute input position
                ih = oh * stride_height - padding_height + kh * dilation_height
                iw = ow * stride_width - padding_width + kw * dilation_width

                # Check bounds
                ih_mask = (ih >= 0) & (ih < input_height)
                iw_mask = (iw >= 0) & (iw < input_width)

                # Input pointer
                input_offset_base = pid_b * in_channels_per_group * groups * input_height * input_width
                input_offset_base += pid_g * in_channels_per_group * input_height * input_width
                input_offset_base += ic * input_height * input_width
                input_offset = input_offset_base + ih * input_width + iw

                x_ptrs = x_ptr + input_offset
                x = tl.load(x_ptrs, mask=ih_mask & iw_mask & (tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels_per_group - ic), other=0.0)

                # Weight pointer
                weight_offset_base = pid_g * (out_channels // groups) * in_channels_per_group * kernel_height * kernel_width
                weight_offset_base += tl.arange(0, BLOCK_SIZE_N)[:, None] * in_channels_per_group * kernel_height * kernel_width
                weight_offset_base += kh * kernel_width * in_channels_per_group
                weight_offset_base += kw * in_channels_per_group
                weight_offset_base += ic
                weight_ptrs = weight_ptr + weight_offset_base + tl.arange(0, BLOCK_SIZE_M)[None, :] * in_channels_per_group

                w = tl.load(weight_ptrs, mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < out_channels // groups) &
                                      (tl.arange(0, BLOCK_SIZE_M)[None, :] < in_channels_per_group - ic), other=0.0)

                # Accumulate
                acc += tl.dot(w, x.to(tl.float32), out_dtype=tl.float32)

    # Add bias
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + pid_g * (out_channels // groups) + tl.arange(0, BLOCK_SIZE_N)
        b = tl.load(bias_ptrs, mask=tl.arange(0, BLOCK_SIZE_N) < out_channels // groups, other=0.0)
        acc += b[:, None]

    # Store output
    mask = (tl.arange(0, BLOCK_SIZE_N)[:, None] < out_channels // groups) & (tl.arange(0, BLOCK_SIZE_M)[None, :] < in_channels_per_group)
    tl.store(out_ptrs, acc, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                            stride: tuple, padding: tuple, dilation: tuple, groups: int):
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    in_channels_per_group = in_channels // groups

    # Calculate output spatial dimensions
    out_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_height - 1) + 1
    out_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_width - 1) + 1

    # Output tensor
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Grid configuration
    def grid(meta):
        return (batch_size, out_height, out_width, groups)

    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, out_channels, out_height, out_width,
        in_channels_per_group, input_height, input_width,
        kernel_height, kernel_width,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=16
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)