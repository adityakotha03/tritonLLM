import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    bias_ptr,
    batch, in_channels, height_in, width_in,
    out_channels, height_out, width_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    groups,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_cg, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    bias_stride_c,
    has_bias: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute starting indices
    batch_start = pid_b * BLOCK_SIZE_BATCH
    c_start = pid_c * BLOCK_SIZE_OUT_CH
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # Offsets within blocks
    b_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_OUT_CH)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)

    # Masks
    b_mask = b_offsets < batch
    c_mask = c_offsets < out_channels
    h_mask = h_offsets < height_out
    w_mask = w_offsets < width_out

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    group_size = in_channels // groups
    out_c_per_group = out_channels // groups

    for group in range(groups):
        weight_off_c_base = group * group_size
        weight_off_k_base = group * out_c_per_group

        for ih in range(kernel_h):
            for iw in range(kernel_w):
                # Compute input spatial location
                h_im = h_offsets - padding_h + ih * dilation_h
                w_im = w_offsets - padding_w + iw * dilation_w

                # Stride and output padding adjustment
                h_im = h_im // stride_h
                w_im = w_im // stride_w
                h_im_mask = (h_im >= 0) & (h_im < height_in) & ((h_offsets - padding_h + ih * dilation_h) % stride_h == 0)
                w_im_mask = (w_im >= 0) & (w_im < width_in) & ((w_offsets - padding_w + iw * dilation_w) % stride_w == 0)

                # Combine masks
                mask = b_mask[:, None, None, None] & c_mask[None, :, None, None] & h_im_mask[None, None, :, None] & w_im_mask[None, None, None, :]
                mask = mask & (h_offsets[None, None, :, None] < height_out) & (w_offsets[None, None, None, :] < width_out)

                # Input pointer offsets
                input_ptrs = input_ptr + \
                    (b_offsets[:, None, None, None] * input_stride_b + \
                     (weight_off_c_base + tl.arange(0, group_size)[None, None, None, :]) * input_stride_c + \
                     h_im[None, None, :, None] * input_stride_h + \
                     w_im[None, None, None, :] * input_stride_w)[mask]

                # Weight pointers
                weight_ptrs = weight_ptr + \
                    (weight_off_k_base + c_offsets[None, :, None, None]) * weight_stride_k + \
                    (tl.arange(0, group_size)[None, None, None, :]) * weight_stride_cg + \
                    ih * weight_stride_h + iw * weight_stride_w
                weight_ptrs = weight_ptrs + 0  # resolve broadcast

                # Load input and weight
                input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
                weight_vals = tl.load(weight_ptrs, mask=c_mask[None, :, None, None] & (tl.arange(0, group_size)[None, None, None, :] < group_size), other=0.0)

                # Reshape and multiply
                input_vals = tl.reshape(input_vals, (BLOCK_SIZE_BATCH, group_size, BLOCK_SIZE_H, BLOCK_SIZE_W))
                weight_vals = tl.reshape(weight_vals, (out_channels // groups, group_size))
                # Perform outer product and accumulate
                for b in range(BLOCK_SIZE_BATCH):
                    for h in range(BLOCK_SIZE_H):
                        for w in range(BLOCK_SIZE_W):
                            if b_offsets[b] < batch and h_offsets[h] < height_out and w_offsets[w] < width_out:
                                input_row = tl.load(input_ptr + b_offsets[b] * input_stride_b +
                                                    tl.arange(0, group_size) * input_stride_c +
                                                    h_im[h] * input_stride_h + w_im[w] * input_stride_w,
                                                    mask=h_im_mask[h] & w_im_mask[w], other=0.0)
                                weight_block = tl.load(weight_ptrs + 0, mask=c_mask[:, None] & (tl.arange(0, group_size)[None, :] < group_size), other=0.0)
                                acc[b, :, h, w] += tl.dot(weight_block, input_row.to(tl.float32))
    # Add bias
    if has_bias:
        bias_vals = tl.load(bias_ptr + c_offsets * bias_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        acc += bias_vals[None, :, None, None]

    # Store output
    output_ptrs = output_ptr + \
        (b_offsets[:, None, None, None] * output_stride_b +
         c_offsets[None, :, None, None] * output_stride_c +
         h_offsets[None, None, :, None] * output_stride_h +
         w_offsets[None, None, None, :] * output_stride_w)
    output_ptrs = output_ptrs + 0  # resolve broadcast
    tl.store(output_ptrs, acc, mask=b_mask[:, None, None, None] & c_mask[None, :, None, None] & h_mask[None, None, :, None] & w_mask[None, None, None, :])


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    dilation: tuple,
    groups: int
):
    batch, in_channels, height_in, width_in = x.shape
    out_channels, weight_in_c_per_group, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    output_pad_h, output_pad_w = output_padding
    dilation_h, dilation_w = dilation

    # Compute output spatial dimensions
    height_out = (height_in - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1
    width_out = (width_in - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1

    # Output tensor
    out = torch.empty((batch, out_channels, height_out, width_out), device=x.device, dtype=x.dtype)

    # Strides
    input_strides = x.stride()
    weight_strides = weight.stride()
    output_strides = out.stride()
    bias_stride_c = bias.stride(0) if bias is not None else 0

    # Define block sizes
    BLOCK_SIZE_BATCH = triton.next_power_of_2(batch)
    BLOCK_SIZE_BATCH = min(max(BLOCK_SIZE_BATCH, 1), 8)
    BLOCK_SIZE_OUT_CH = triton.next_power_of_2(out_channels)
    BLOCK_SIZE_OUT_CH = min(max(BLOCK_SIZE_OUT_CH, 1), 32)
    BLOCK_SIZE_H = min(triton.next_power_of_2(height_out), 32)
    BLOCK_SIZE_W = min(triton.next_power_of_2(width_out), 32)

    # Grid
    grid = (
        triton.cdiv(batch, BLOCK_SIZE_BATCH),
        triton.cdiv(out_channels, BLOCK_SIZE_OUT_CH),
        triton.cdiv(height_out, BLOCK_SIZE_H),
        triton.cdiv(width_out, BLOCK_SIZE_W)
    )

    # Launch kernel
    _conv_transpose2d_kernel[grid](
        x, weight, out, bias,
        batch, in_channels, height_in, width_in,
        out_channels, height_out, width_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        output_pad_h, output_pad_w,
        dilation_h, dilation_w,
        groups,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        bias_stride_c,
        has_bias=bias is not None,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_OUT_CH=BLOCK_SIZE_OUT_CH,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias_enabled = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels // groups, *kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.dilation, self.groups
        )