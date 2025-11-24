import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, out_channels, input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_depth, kernel_height, kernel_width,
    stride_depth, stride_height, stride_width,
    padding_depth, padding_height, padding_width,
    output_padding_depth, output_padding_height, output_padding_width,
    groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_cg, weight_stride_d, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_k, output_stride_d, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_IN_CHANNEL: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2) // (output_height * output_width // BLOCK_SIZE_HW)
    pid_hw = tl.program_id(axis=2) % (output_height * output_width // BLOCK_SIZE_HW)

    # Compute output spatial indices
    hw_block_start = pid_hw * BLOCK_SIZE_HW
    d_offset = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)[:, None]
    hw_offset = hw_block_start + tl.arange(0, BLOCK_SIZE_HW)[None, :]

    d_mask = d_offset < output_depth
    hw_mask = hw_offset < output_height * output_width
    mask = d_mask and hw_mask

    # Row-major to spatial (d, h, w)
    d = d_offset
    h = (hw_offset // output_width) % output_height
    w = hw_offset % output_width

    # Transposed conv: output += sum_{c, kd, kh, kw} input[b, c, id, ih, iw] * weight[k, c, kd, kh, kw]
    # where id = (d - kd - padding_d) / stride_d, etc.
    group_id = pid_k // (out_channels // groups)
    out_c_per_group = out_channels // groups
    c_offset = group_id * (in_channels // groups) + tl.arange(0, BLOCK_SIZE_IN_CHANNEL)
    c_mask = c_offset < (group_id + 1) * (in_channels // groups)

    kd = tl.arange(0, kernel_depth)
    kh = tl.arange(0, kernel_height)
    kw = tl.arange(0, kernel_width)

    # Compute input indices
    id = (d[:, None] - padding_depth - kd[None, :]) / stride_depth
    ih = (h[:, None] - padding_height - kh[None, :]) / stride_height
    iw = (w[:, None] - padding_width - kw[None, :]) / stride_width

    # Check if input indices are valid and divisible
    valid_id = (id >= 0) & (id < input_depth) & ((d[:, None] - padding_depth - kd[None, :]) % stride_depth == 0)
    valid_ih = (ih >= 0) & (ih < input_height) & ((h[:, None] - padding_height - kh[None, :]) % stride_height == 0)
    valid_iw = (iw >= 0) & (iw < input_width) & ((w[:, None] - padding_width - kw[None, :]) % stride_width == 0)
    valid_input = valid_id[:, None, None, :] & valid_ih[:, :, None, None] & valid_iw[:, None, :, None]

    id = id.to(tl.int32)
    ih = ih.to(tl.int32)
    iw = iw.to(tl.int32)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_HW), dtype=tl.float32)

    # Iterate over input channels in blocks
    for c in range(0, in_channels // groups, BLOCK_SIZE_IN_CHANNEL):
        c_block_start = c_offset[0]
        if c_block_start >= (in_channels // groups):
            break
        c_mask_curr = c_mask

        # Load input tiles: [BLOCK_SIZE_D, BLOCK_SIZE_HW, BLOCK_SIZE_IN_CHANNEL, kernel_depth, kernel_height, kernel_width]
        input_vals = tl.load(
            input_ptr +
            pid_b * input_stride_b +
            c_block_start * input_stride_c +
            id[:, :, None, None, None] * input_stride_d +
            ih[:, :, :, None, None] * input_stride_h +
            iw[:, :, None, :, None] * input_stride_w,
            mask=valid_input[None, :, :, :, :] & c_mask_curr[None, None, None, None, None, :],
            other=0.0
        )  # Shape: [BLOCK_SIZE_D, kernel_depth, kernel_height, kernel_width, BLOCK_SIZE_IN_CHANNEL]

        # Reshape input_vals to match weight shape
        input_vals = tl.reshape(input_vals, (BLOCK_SIZE_D, BLOCK_SIZE_HW, kernel_depth, kernel_height, kernel_width, BLOCK_SIZE_IN_CHANNEL))
        input_vals = tl.transpose(input_vals, 2, 5)  # -> [BLOCK_SIZE_D, BLOCK_SIZE_HW, BLOCK_SIZE_IN_CHANNEL, kernel_depth, kernel_height, kernel_width]

        # Load weights: [out_c_per_group, BLOCK_SIZE_IN_CHANNEL, kernel_depth, kernel_height, kernel_width]
        weight_vals = tl.load(
            weight_ptr +
            (pid_k % out_c_per_group) * weight_stride_k +
            c_block_start * weight_stride_cg +
            kd[None, :, None, None] * weight_stride_d +
            kh[None, None, :, None] * weight_stride_h +
            kw[None, None, None, :] * weight_stride_w,
            mask=c_mask_curr[None, :, None, None, None],
            other=0.0
        )  # [out_c_per_group, BLOCK_SIZE_IN_CHANNEL, kernel_depth, kernel_height, kernel_width]

        # Reshape weight_vals to broadcast with input_vals
        weight_vals = weight_vals[None, None, :, :, :, :]  # [1, 1, out_c_per_group, BLOCK_SIZE_IN_CHANNEL, kernel_depth, kernel_height, kernel_width]

        # Perform outer product and sum-reduction
        product = input_vals[None, :, :, :, :, :, :] * weight_vals  # Broadcasting
        local_sum = tl.sum(product, axis=[3, 4, 5, 6])  # Sum over in_channel and kernel dims

        acc += tl.sum(local_sum, axis=0)  # Accumulate over input channel block

    # Write back result
    output_offset = (
        pid_b * output_stride_b +
        pid_k * output_stride_k +
        d * output_stride_d +
        h * output_stride_h +
        w * output_stride_w
    )
    tl.store(output_ptr + output_offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight and optional bias
        kernel_depth, kernel_height, kernel_width = kernel_size
        weight_shape = (in_channels, out_channels // groups, kernel_depth, kernel_height, kernel_width)
        self.weight = nn.Parameter(torch.empty(*weight_shape))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias:
            nn.init.zeros_(self.bias_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        kernel_depth, kernel_height, kernel_width = self.kernel_size
        stride_depth, stride_height, stride_width = self.stride
        padding_depth, padding_height, padding_width = self.padding
        output_padding_depth, output_padding_height, output_padding_width = self.output_padding

        # Compute output shape
        output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth
        output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height
        output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width

        # Output tensor
        out = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()

        # Grid
        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
                triton.cdiv(self.out_channels, meta['BLOCK_SIZE_OUT_CHANNEL']),
                triton.cdiv(output_depth * output_height * output_width, meta['BLOCK_SIZE_HW'])
            )

        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, self.weight, out,
            batch_size, in_channels, self.out_channels,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width,
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width,
            output_padding_depth, output_padding_height, output_padding_width,
            self.groups,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
            BLOCK_SIZE_BATCH=1,
            BLOCK_SIZE_OUT_CHANNEL=16,
            BLOCK_SIZE_IN_CHANNEL=16,
            BLOCK_SIZE_D=4,
            BLOCK_SIZE_HW=64
        )

        # Add bias if needed
        if self.bias:
            out += self.bias_param.view(1, -1, 1, 1, 1)

        return out