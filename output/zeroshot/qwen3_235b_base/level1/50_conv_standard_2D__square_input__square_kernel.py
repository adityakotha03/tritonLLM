import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, out_channels, out_height, out_width,
    in_channels, in_height, in_width,
    kernel_size_h, kernel_size_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_IN_CH: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute starting indices
    batch_start = pid_b * BLOCK_SIZE_BATCH
    ch_out_start = pid_c * BLOCK_SIZE_OUT_CH
    hw_start = pid_hw * BLOCK_SIZE_HW

    # Offsets within blocks
    b_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    ch_out_offsets = ch_out_start + tl.arange(0, BLOCK_SIZE_OUT_CH)
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)

    # Masks
    b_mask = b_offsets < batch
    hw_mask = hw_offsets < out_height * out_width
    ch_out_mask = ch_out_offsets < out_channels

    # 2D to 1D indices
    ohw = hw_offsets // out_width
    oww = hw_offsets % out_width
    oh = ohw
    ow = oww

    # Input spatial indices
    ih = oh * stride_h - padding_h
    iw = ow * stride_w - padding_w

    # Weight layout: (out_channels, in_channels, kh, kw)
    # Input layout: (batch, in_channels, height, width)
    # Output layout: (batch, out_channels, out_h, out_w)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_HW), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for ch_in_base in range(0, in_channels, BLOCK_SIZE_IN_CH):
        for kh_base in range(0, kernel_size_h, BLOCK_SIZE_K):
            for kw_base in range(0, kernel_size_w, BLOCK_SIZE_K):
                ch_in_offsets = ch_in_base + tl.arange(0, BLOCK_SIZE_IN_CH)
                kh_offsets = kh_base + tl.arange(0, BLOCK_SIZE_K)
                kw_offsets = kw_base + tl.arange(0, BLOCK_SIZE_K)

                ch_in_mask = ch_in_offsets < in_channels
                kh_mask = kh_offsets < kernel_size_h
                kw_mask = kw_offsets < kernel_size_w

                # Input indices with dilation
                ih_exp = ih[:, None] + kh_offsets[None, :] * dilation_h
                iw_exp = iw[:, None] + kw_offsets[None, :] * dilation_w

                # Bounds checking for input
                ih_valid = (ih_exp >= 0) & (ih_exp < in_height)
                iw_valid = (iw_exp >= 0) & (iw_exp < in_width)
                valid = ih_valid & iw_valid

                # Broadcast masks
                mask = b_mask[:, None, None] & ch_out_mask[None, :, None] & ch_in_mask[None, None, :] & \
                       hw_mask[:, None, None] & kh_mask[None, None, :] & kw_mask[None, None, :] & valid[:, None, :, :]

                # Load input: (BLOCK_SIZE_BATCH, BLOCK_SIZE_IN_CH, BLOCK_SIZE_HW, BLOCK_SIZE_K, BLOCK_SIZE_K)
                input_vals = tl.load(
                    input_ptr +
                    b_offsets[:, None, None, None, None] * input_stride_b +
                    ch_in_offsets[None, :, None, None, None] * input_stride_c +
                    ih_exp[:, None, :, None, None] * input_stride_h +
                    iw_exp[:, None, None, :, None] * input_stride_w,
                    mask=mask,
                    other=0.0
                )

                # Load weights: (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH, BLOCK_SIZE_K, BLOCK_SIZE_K)
                weight_vals = tl.load(
                    weight_ptr +
                    ch_out_offsets[:, None, None, None] * weight_stride_c +
                    ch_in_offsets[None, :, None, None] * weight_stride_k +
                    kh_offsets[None, None, :, None] * weight_stride_h +
                    kw_offsets[None, None, None, :] * weight_stride_w,
                    mask=ch_out_mask[:, None, None, None] &
                         ch_in_mask[None, :, None, None] &
                         kh_mask[None, None, :, None] &
                         kw_mask[None, None, None, :],
                    other=0.0
                )

                # Reshape for matmul: (BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_HW) @ (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH, BLOCK_SIZE_K, BLOCK_SIZE_K)
                # -> (BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_HW)
                weight_vals = weight_vals[None, :, :, :, :]  # Add batch dim
                input_vals = input_vals.permute(0, 2, 3, 4, 1)  # -> (B, HW, K, K, C_in)
                weight_vals = weight_vals.permute(0, 1, 3, 4, 2)  # -> (1, C_out, K, K, C_in)
                input_vals = input_vals.reshape(BLOCK_SIZE_BATCH, BLOCK_SIZE_HW, -1)
                weight_vals = weight_vals.reshape(BLOCK_SIZE_OUT_CH, -1)
                product = tl.dot(input_vals, weight_vals.T)
                product = product.reshape(BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_HW)

                acc += product

    # Store output
    output_offsets = \
        b_offsets[:, None, None] * output_stride_b + \
        ch_out_offsets[None, :, None] * output_stride_c + \
        hw_offsets[None, None, :] * output_stride_w
    output_offsets = output_offsets + (hw_offsets // out_width)[None, None, :] * output_stride_h
    tl.store(output_ptr + output_offsets, acc, mask=b_mask[:, None, None] & ch_out_mask[None, :, None] & hw_mask[None, None, :])


def triton_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert groups == 1, "Grouped conv not supported"
    assert dilation == 1 or (isinstance(dilation, tuple) and dilation[0] == 1 and dilation[1] == 1), "Dilation not supported"
    assert bias is None, "Bias not supported in this kernel"

    batch, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding

    out_height = (in_height + 2 * padding_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation * (kernel_w - 1) - 1) // stride_w + 1

    # Output tensor
    output = torch.empty((batch, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)

    # Define block sizes
    BLOCK_SIZE_BATCH = triton.next_power_of_2(batch)
    BLOCK_SIZE_BATCH = min(BLOCK_SIZE_BATCH, 8)
    BLOCK_SIZE_OUT_CH = 16
    BLOCK_SIZE_HW = 64
    BLOCK_SIZE_IN_CH = 16
    BLOCK_SIZE_K = 4

    # Grid
    grid = (
        triton.cdiv(batch, BLOCK_SIZE_BATCH),
        triton.cdiv(out_channels, BLOCK_SIZE_OUT_CH),
        triton.cdiv(out_height * out_width, BLOCK_SIZE_HW)
    )

    # Strides
    input_stride_b, input_stride_c, input_stride_h, input_stride_w = input.stride()
    weight_stride_c, weight_stride_k, weight_stride_h, weight_stride_w = weight.stride()
    output_stride_b, output_stride_c, output_stride_h, output_stride_w = output.stride()

    conv2d_kernel[grid](
        input, weight, output,
        batch, out_channels, out_height, out_width,
        in_channels, in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation, dilation,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        weight_stride_c, weight_stride_k, weight_stride_h, weight_stride_w,
        output_stride_b, output_stride_c, output_stride_h, output_stride_w,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_OUT_CH=BLOCK_SIZE_OUT_CH,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_IN_CH=BLOCK_SIZE_IN_CH,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.zeros(96))
        self.stride = 4
        self.padding = 2

    def forward(self, x):
        x = triton_conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        return x