import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    bias_ptr,
    batch_size, in_channels, out_channels, input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_size_d, kernel_size_h, kernel_size_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    output_padding_d, output_padding_h, output_padding_w,
    groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_g, weight_stride_r, weight_stride_s, weight_stride_t,
    output_stride_b, output_stride_k, output_stride_d, output_stride_h, output_stride_w,
    bias_ptr_stride,
    has_bias: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_DEPTH: tl.constexpr,
    BLOCK_SIZE_HEIGHT: tl.constexpr,
    BLOCK_SIZE_WIDTH: tl.constexpr,
    BLOCK_SIZE_KERNEL_D: tl.constexpr,
    BLOCK_SIZE_KERNEL_H: tl.constexpr,
    BLOCK_SIZE_KERNEL_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)
    pid_h = tl.program_id(axis=3)
    pid_w = tl.program_id(axis=4)

    # Compute starting indices
    batch_start = pid_b * BLOCK_SIZE_BATCH
    out_ch_start = pid_k * BLOCK_SIZE_OUT_CHANNEL
    out_d_start = pid_d * BLOCK_SIZE_DEPTH
    out_h_start = pid_h * BLOCK_SIZE_HEIGHT
    out_w_start = pid_w * BLOCK_SIZE_WIDTH

    # Ranges
    batch_range = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    out_ch_range = out_ch_start + tl.arange(0, BLOCK_SIZE_OUT_CHANNEL)
    out_d_range = out_d_start + tl.arange(0, BLOCK_SIZE_DEPTH)
    out_h_range = out_h_start + tl.arange(0, BLOCK_SIZE_HEIGHT)
    out_w_range = out_w_start + tl.arange(0, BLOCK_SIZE_WIDTH)

    # Masks
    batch_mask = batch_range < batch_size
    out_ch_mask = out_ch_range < out_channels
    out_d_mask = out_d_range < output_depth
    out_h_mask = out_h_range < output_height
    out_w_mask = out_w_range < output_width

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNEL, BLOCK_SIZE_DEPTH, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH),
                   dtype=tl.float32)

    # Group parameters
    channels_per_group = in_channels // groups
    group_id = out_ch_start // (out_channels // groups)

    # Iterate over input channels in group
    for ic in range(0, channels_per_group, BLOCK_SIZE_KERNEL_W):
        ic_block = ic + tl.arange(0, BLOCK_SIZE_KERNEL_W)
        ic_mask = ic_block < channels_per_group
        global_ic = group_id * channels_per_group + ic_block
        input_ch_mask = global_ic < in_channels

        # Iterate over kernel
        for kd in range(0, kernel_size_d):
            for kh in range(0, kernel_size_h):
                for kw in range(0, kernel_size_w):
                    # Compute input location
                    in_d = out_d_range - padding_d + kd * dilation_d
                    in_h = out_h_range - padding_h + kh * dilation_h
                    in_w = out_w_range - padding_w + kw * dilation_w

                    # Stride
                    in_d = in_d // stride_d
                    in_h = in_h // stride_h
                    in_w = in_w // stride_w

                    # Check bounds
                    in_d_valid = (in_d >= 0) & (in_d < input_depth)
                    in_h_valid = (in_h >= 0) & (in_h < input_height)
                    in_w_valid = (in_w >= 0) & (in_w < input_width)

                    # Combine masks
                    valid = batch_mask[:, None, None, None, None] & \
                            out_ch_mask[None, :, None, None, None] & \
                            in_d_valid[None, None, :, None, None] & \
                            in_h_valid[None, None, None, :, None] & \
                            in_w_valid[None, None, None, None, :] & \
                            ic_mask[None, None, None, None, :]

                    # Load input: [B, C_in, D, H, W]
                    input_offsets = \
                        batch_range[:, None, None, None, None] * input_stride_b + \
                        global_ic[None, None, None, None, :] * input_stride_c + \
                        in_d[None, None, :, None, None] * input_stride_d + \
                        in_h[None, None, None, :, None] * input_stride_h + \
                        in_w[None, None, None, None, :] * input_stride_w
                    input_vals = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)

                    # Load weights: [out_ch, group, kd, kh, kw]
                    weight_offsets = \
                        out_ch_range[:, None] * weight_stride_k + \
                        group_id * weight_stride_g + \
                        kd * weight_stride_r + \
                        kh * weight_stride_s + \
                        kw * weight_stride_t + \
                        ic_block[None, :] * weight_stride_c if hasattr(tl, 'weight_stride_c') else 0
                    # Note: Triton doesn't support dynamic strides; we assume contiguous layout
                    # Instead, we recompute flat offset
                    weight_flat_offset = \
                        out_ch_range[:, None] * (groups * kernel_size_d * kernel_size_h * kernel_size_w * channels_per_group) + \
                        group_id * (kernel_size_d * kernel_size_h * kernel_size_w * channels_per_group) + \
                        kd * (kernel_size_h * kernel_size_w * channels_per_group) + \
                        kh * (kernel_size_w * channels_per_group) + \
                        kw * (channels_per_group) + \
                        ic_block[None, :]
                    weight_vals = tl.load(weight_ptr + weight_flat_offset, mask=out_ch_mask[:, None] & ic_mask[None, :], other=0.0)

                    # Multiply and accumulate
                    # input_vals: [B, 1, D, H, W, C_block] -> [B, 1, D, H, W, C_block]
                    # weight_vals: [K, C_block] -> [1, K, 1, 1, 1, C_block]
                    weight_vals = weight_vals[:, None, None, None, :]  # Reshape for broadcasting
                    product = input_vals[None, :, :, :, :, :] * weight_vals[:, None, :, :, :, :]
                    acc += tl.sum(product, axis=5)  # Sum over input channel block

    # Add bias if present
    if has_bias:
        bias_vals = tl.load(bias_ptr + out_ch_range, mask=out_ch_mask, other=0.0)
        acc += bias_vals[:, None, None, None, None]

    # Store output
    output_offsets = \
        batch_range[:, None, None, None, None] * output_stride_b + \
        out_ch_range[None, :, None, None, None] * output_stride_k + \
        out_d_range[None, None, :, None, None] * output_stride_d + \
        out_h_range[None, None, None, :, None] * output_stride_h + \
        out_w_range[None, None, None, None, :] * output_stride_w
    output_mask = batch_mask[:, None, None, None, None] & \
                  out_ch_mask[None, :, None, None, None] & \
                  out_d_mask[None, None, :, None, None] & \
                  out_h_mask[None, None, None, :, None] & \
                  out_w_mask[None, None, None, None, :]
    tl.store(output_ptr + output_offsets, acc, mask=output_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    output_padding,
    dilation,
):
    batch_size, in_channels, input_depth, input_height, input_width = x.shape
    out_channels, groups, kernel_size_d, kernel_size_h, kernel_size_w = weight.shape
    groups = 1  # Simplifying assumption: groups=1

    # Compute output shape
    def transposed_conv_output_length(input_length, output_padding, stride, padding, dilation, kernel_size):
        return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    output_depth = transposed_conv_output_length(input_depth, output_padding[0], stride[0], padding[0], dilation[0], kernel_size_d)
    output_height = transposed_conv_output_length(input_height, output_padding[1], stride[1], padding[1], dilation[1], kernel_size_h)
    output_width = transposed_conv_output_length(input_width, output_padding[2], stride[2], padding[2], dilation[2], kernel_size_w)

    # Output tensor
    out = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, dtype=torch.float32, device=x.device)

    # Strides
    input_strides = x.stride()
    weight_strides = weight.stride()
    output_strides = out.stride()

    # Grid
    def grid(meta):
        return (
            triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
            triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT_CHANNEL']),
            triton.cdiv(output_depth, meta['BLOCK_SIZE_DEPTH']),
            triton.cdiv(output_height, meta['BLOCK_SIZE_HEIGHT']),
            triton.cdiv(output_width, meta['BLOCK_SIZE_WIDTH']),
        )

    # Launch kernel
    _conv_transpose3d_kernel[grid](
        x, weight, out, bias,
        batch_size, in_channels, out_channels, input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
        bias.stride(0) if bias is not None else 0,
        has_bias=bias is not None,
        BLOCK_SIZE_BATCH=1,
        BLOCK_SIZE_OUT_CHANNEL=16,
        BLOCK_SIZE_DEPTH=8,
        BLOCK_SIZE_HEIGHT=8,
        BLOCK_SIZE_WIDTH=8,
        BLOCK_SIZE_KERNEL_D=3,
        BLOCK_SIZE_KERNEL_H=3,
        BLOCK_SIZE_KERNEL_W=16,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized version of transposed 3D convolution using Triton.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.use_bias = bias

        # Initialize weight and bias
        kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *kernel_size_tuple
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the same device as parameters
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            if self.bias is not None:
                self.bias = self.bias.to(x.device)

        # Call Triton kernel
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.dilation
        )