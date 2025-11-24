import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    batch_size, in_channels, input_height, input_width,
    output_height, output_width, kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_kh, weight_stride_kw,
    output_stride_b, output_stride_c, output_stride_oh, output_stride_ow,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CH: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute program ids
    pid_b = tl.program_id(axis=0)
    pid_ch = tl.program_id(axis=1)
    pid_oh = tl.program_id(axis=2)
    pid_ow = tl.program_id(axis=3)

    # Compute offsets for output tile
    batch_start = pid_b * BLOCK_SIZE_BATCH
    ch_start = pid_ch * BLOCK_SIZE_CH
    oh_start = pid_oh * BLOCK_SIZE_H
    ow_start = pid_ow * BLOCK_SIZE_W

    # Load input and weight tiles
    # Output coordinates
    oh_offsets = oh_start + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_SIZE_W)
    mask_oh = oh_offsets < output_height
    mask_ow = ow_offsets < output_width
    mask_out = mask_oh[:, None] & mask_ow[None, :]

    # Input spatial start positions
    ih_start = oh_offsets * stride_h - padding_h
    iw_start = ow_offsets * stride_w - padding_w

    # Weight offsets
    kh_offsets = tl.arange(0, kernel_h)
    kw_offsets = tl.arange(0, kernel_w)

    # Broadcast to [BLOCK_SIZE_H, BLOCK_SIZE_W, kernel_h, kernel_w]
    ih = ih_start[:, None, None, None] + dilation_h * kh_offsets[None, None, :, None]
    iw = iw_start[None, :, None, None] + dilation_w * kw_offsets[None, None, None, :]
    ch = ch_start + tl.arange(0, BLOCK_SIZE_CH)

    # Input mask
    mask_ih = (ih >= 0) & (ih < input_height)
    mask_iw = (iw >= 0) & (iw < input_width)
    mask_input = mask_ih & mask_iw

    # Weight mask (no mask needed if kernel is full)
    mask_weight = (
        (kh_offsets[None, None, :, None] < kernel_h) &
        (kw_offsets[None, None, None, :] < kernel_w) &
        (ch < in_channels)
    )

    # Broadcast masks
    mask = mask_input & mask_weight & mask_out[:, :, None, None]

    # Expand indices
    input_indices = (
        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH)[:, None, None, None]) * input_stride_b +
        ch[None, :, None, None] * input_stride_c +
        ih * input_stride_h +
        iw * input_stride_w
    )
    weight_indices = (
        ch[None, :, None, None] * weight_stride_c +
        kh_offsets[None, None, :, None] * weight_stride_kh +
        kw_offsets[None, None, None, :] * weight_stride_kw
    )

    # Load input and weights
    x = tl.load(x_ptr + input_indices, mask=mask, other=0.0)
    w = tl.load(w_ptr + weight_indices, mask=mask_weight, other=0.0)

    # Perform convolution: sum over kh, kw
    w = w[None, :, :, :]  # Add batch dim
    output = tl.sum(x * w, axis=3)  # kw
    output = tl.sum(output, axis=2)  # kh

    # Store output
    output_indices = (
        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH)[:, None, None]) * output_stride_b +
        ch[None, :, None] * output_stride_c +
        oh_offsets[None, None, :] * output_stride_oh +
        ow_offsets[None, :, None] * output_stride_ow
    )
    tl.store(y_ptr + output_indices, output)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == in_channels and out_channels == in_channels, "Only depthwise with in_channels == out_channels and groups=in_channels supported"
        self.in_channels = in_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))

        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()

        # Get shapes
        batch_size, in_channels, input_height, input_width = x.shape
        kernel_h, kernel_w = self.kernel_size_h, self.kernel_size_w
        output_height = (input_height + 2 * self.padding_h - dilation_h * (kernel_h - 1) - 1) // self.stride_h + 1
        output_width = (input_width + 2 * self.padding_w - dilation_w * (kernel_w - 1) - 1) // self.stride_w + 1

        # Output tensor
        output = torch.empty(batch_size, self.in_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = output.stride()

        # Launch kernel
        def grid(META):
            return (
                triton.cdiv(batch_size, META['BLOCK_SIZE_BATCH']),
                triton.cdiv(in_channels, META['BLOCK_SIZE_CH']),
                triton.cdiv(output_height, META['BLOCK_SIZE_H']),
                triton.cdiv(output_width, META['BLOCK_SIZE_W']),
            )

        depthwise_conv2d_kernel[grid](
            x, self.weight, output,
            batch_size, in_channels, input_height, input_width,
            output_height, output_width, kernel_h, kernel_w,
            self.stride_h, self.stride_w, self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3],
            weight_strides[0], weight_strides[1], weight_strides[2],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3],
            BLOCK_SIZE_BATCH=1,
            BLOCK_SIZE_CH=16,
            BLOCK_SIZE_H=16,
            BLOCK_SIZE_W=32,
        )

        # Add bias if needed
        if self.bias:
            output += self.bias_param.view(1, -1, 1, 1)

        return output