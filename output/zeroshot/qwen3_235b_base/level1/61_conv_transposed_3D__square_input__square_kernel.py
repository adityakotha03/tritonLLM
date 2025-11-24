import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    bias_ptr,
    batch_size, in_channels, out_channels, input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_size, stride, padding, output_padding,
    groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_d, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    bias_stride_c,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_od = tl.program_id(axis=1)
    pid_ohw = tl.program_id(axis=2)
    pid_g = tl.program_id(axis=3)

    # Calculate output spatial indices
    output_hw = pid_ohw
    oh = (output_hw // output_width) % output_height
    ow = output_hw % output_width
    od = pid_od

    # Initialize pointers for output
    output_offset = (
        pid_b * output_stride_b +
        pid_g * (out_channels // groups) * output_stride_c +
        od * output_stride_d + oh * output_stride_h + ow * output_stride_w
    )
    output_ptrs = output_ptr + output_offset + tl.arange(0, BLOCK_SIZE_C)[:, None] * output_stride_c + tl.arange(0, BLOCK_SIZE_HW)[None, :]

    # Load bias if exists
    bias_offset = pid_g * (out_channels // groups) * bias_stride_c + tl.arange(0, BLOCK_SIZE_C) * bias_stride_c
    bias = tl.load(bias_ptr + bias_offset, mask=tl.arange(0, BLOCK_SIZE_C) < (out_channels // groups), other=0.0)

    # Accumulate result
    acc = bias[:, None]

    # Loop over input channels and kernel
    for ic_group in range(in_channels // groups):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Compute input position
                    id = od * stride - padding + kd
                    ih = oh * stride - padding + kh
                    iw = ow * stride - padding + kw

                    # Check bounds
                    id_mask = (id >= 0) & (id < input_depth)
                    ih_mask = (ih >= 0) & (ih < input_height)
                    iw_mask = (iw >= 0) & (iw < input_width)
                    mask = id_mask & ih_mask & iw_mask

                    # Input pointer
                    input_offset = (
                        pid_b * input_stride_b +
                        (pid_g * (in_channels // groups) + ic_group) * input_stride_c +
                        id * input_stride_d + ih * input_stride_h + iw * input_stride_w
                    )
                    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

                    # Weight pointer
                    weight_offset = (
                        (pid_g * (out_channels // groups)) * weight_stride_c +
                        (ic_group) * weight_stride_k +
                        kd * weight_stride_d + kh * weight_stride_h + kw * weight_stride_w
                    )
                    weight_vals = tl.load(
                        weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE_C)[:, None] * weight_stride_c,
                        mask=(tl.arange(0, BLOCK_SIZE_C)[:, None] < (out_channels // groups)) &
                             (tl.arange(0, BLOCK_SIZE_K)[None, :] < 1),
                        other=0.0
                    )
                    weight_val = weight_vals[:, 0]

                    # Multiply and accumulate
                    acc += input_val * weight_val[:, None]

    # Store output
    o_mask = (tl.arange(0, BLOCK_SIZE_C)[:, None] < (out_channels // groups)) & \
             (tl.arange(0, BLOCK_SIZE_HW)[None, :] < output_height * output_width)
    tl.store(output_ptrs, acc, mask=o_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dimensions
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        assert in_channels == self.in_channels

        # Output dimensions
        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Output tensor
        out = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()
        bias_stride = self.bias.stride()[0] if self.bias is not None else 0

        # Launch kernel
        def grid(meta):
            return (
                batch_size,
                triton.cdiv(output_depth, meta['BLOCK_SIZE_D']),
                triton.cdiv(output_height * output_width, meta['BLOCK_SIZE_HW']),
                self.groups
            )

        # Set block sizes
        BLOCK_SIZE_D = 16
        BLOCK_SIZE_HW = 64
        BLOCK_SIZE_C = 16
        BLOCK_SIZE_K = 1

        conv_transpose3d_kernel[grid](
            x, self.weight, out, self.bias,
            batch_size, in_channels, self.out_channels,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width,
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.groups,
            *input_strides,
            *weight_strides,
            *output_strides,
            bias_stride,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        return out