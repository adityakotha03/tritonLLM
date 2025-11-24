import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, out_channels, in_channels, depth_out, height_out, width_out, depth_in, height_in, width_in,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_g, weight_stride_r, weight_stride_s, weight_stride_t,
    output_stride_b, output_stride_c, output_stride_z, output_stride_y, output_stride_x,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_CIN: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_y = tl.program_id(2)
    pid_x = tl.program_id(3)
    pid_g = tl.program_id(4)

    # Output spatial offsets
    z = pid_z * BLOCK_D + tl.arange(0, BLOCK_D)
    y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
    x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)

    z_mask = z < depth_out
    y_mask = y < height_out
    x_mask = x < width_out
    xyz_mask = z_mask[:, None, None] & y_mask[:, None] & x_mask

    # Input spatial indices
    # For transposed conv: input index = (output_index + padding - kernel + output_padding) // stride
    # But easier to loop over input and accumulate into output
    # Instead, we reverse: for each output position, loop over contributing input positions and kernel weights

    # We change strategy: for each output location (z, y, x), we loop over kernel and input
    # Input index: i_d = (z - pad_d + s_d * kd - op_d) / s_d
    # But must be integer and within bounds

    # Instead, we use the forward formula:
    # Input location that contributes: d_in = (z - kd * stride_d) // stride_d + padding_d
    # But better: for each output point, we loop over kernel dimensions

    # Let's reframe: for each output point (z, y, x), we sum over:
    #   kd in [0, kernel_d), kh in [0, kernel_h), kw in [0, kernel_w)
    #   d_in = z - kd * stride_d + padding_d
    #   h_in = y - kh * stride_h + padding_h
    #   w_in = x - kw * stride_w + padding_w
    # Then check if d_in, h_in, w_in are in bounds

    # But we need to split output channels by groups
    channels_per_group = out_channels // groups
    out_ch_start = pid_g * channels_per_group
    out_ch_end = out_ch_start + channels_per_group
    out_ch_range = tl.arange(0, BLOCK_K)
    ch_mask = out_ch_range < channels_per_group

    # We will iterate over input channels per group
    input_channels_per_group = in_channels // groups
    for cid in range(0, input_channels_per_group, BLOCK_CIN):
        cin_off = tl.arange(0, BLOCK_CIN)
        cin_mask = cin_off < input_channels_per_group
        full_cin_mask = cin_mask[None, :, None, None, None] & ch_mask[:, None, None, None, None] & xyz_mask

        # Initialize accumulator
        acc = tl.zeros((BLOCK_K, BLOCK_CIN, BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

        # Loop over kernel
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Compute input spatial location
                    d_in = z - kd * stride_d + padding_d
                    h_in = y - kh * stride_h + padding_h
                    w_in = x - kw * stride_w + padding_w

                    # Check bounds
                    d_in_valid = (d_in >= 0) & (d_in < depth_in)
                    h_in_valid = (h_in >= 0) & (h_in < height_in)
                    w_in_valid = (w_in >= 0) & (w_in < width_in)
                    valid = d_in_valid[:, None, None] & h_in_valid[:, None] & w_in_valid
                    valid = valid[None, None, :, :, :]

                    # Load input: [batch, group_input_ch, depth_in, height_in, width_in]
                    input_offset = (
                        pid_b * input_stride_b +
                        (pid_g * input_channels_per_group + cin_off) * input_stride_c +
                        d_in[:, None, None] * input_stride_d +
                        h_in[:, None] * input_stride_h +
                        w_in * input_stride_w
                    )
                    input_vals = tl.load(input_ptr + input_offset, mask=full_cin_mask & valid, other=0.0)

                    # Load weights: [out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w]
                    weight_offset = (
                        (out_ch_start + out_ch_range) * weight_stride_k +
                        cin_off * weight_stride_g +
                        kd * weight_stride_r +
                        kh * weight_stride_s +
                        kw * weight_stride_t
                    )
                    weights = tl.load(weight_ptr + weight_offset, mask=full_cin_mask, other=0.0)

                    # Multiply and accumulate
                    acc += weights[:, :, None, None, None] * input_vals[None, :, :, :, :]

        # Store accumulator to output
        output_offset = (
            pid_b * output_stride_b +
            (out_ch_start + out_ch_range)[:, None, None, None] * output_stride_c +
            z[:, None, None] * output_stride_z +
            y[:, None] * output_stride_y +
            x * output_stride_x
        )
        output_mask = ch_mask[:, None, None, None] & xyz_mask
        tl.store(output_ptr + output_offset, acc, mask=output_mask)


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

        # Initialize weight
        kernel_d, kernel_h, kernel_w = kernel_size
        weight_shape = (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(weight_shape))

        # Initialize bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

        # Initialize weights using kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape
        batch_size, in_channels, depth, height, width = x.shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        op_d, op_h, op_w = self.output_padding

        # Output shape
        depth_out = (depth - 1) * stride_d - 2 * pad_d + kernel_d + op_d
        height_out = (height - 1) * stride_h - 2 * pad_h + kernel_h + op_h
        width_out = (width - 1) * stride_w - 2 * pad_w + kernel_w + op_w

        # Output tensor
        out = torch.empty(batch_size, self.out_channels, depth_out, height_out, width_out, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()

        # Launch kernel
        # Block sizes
        BLOCK_D = 16
        BLOCK_H = 16
        BLOCK_W = 16
        BLOCK_K = 16
        BLOCK_CIN = 16

        # Grid
        grid = (
            batch_size,
            triton.cdiv(depth_out, BLOCK_D),
            triton.cdiv(height_out, BLOCK_H),
            triton.cdiv(width_out, BLOCK_W),
            self.groups
        )

        conv_transpose3d_kernel[grid](
            x, self.weight, out,
            batch_size, self.out_channels, self.in_channels,
            depth_out, height_out, width_out,
            depth, height, width,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            op_d, op_h, op_w,
            self.groups,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
            BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            BLOCK_K=BLOCK_K, BLOCK_CIN=BLOCK_CIN
        )

        # Add bias
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1, 1)

        return out