import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,          # pointer to input tensor (batch, in_channels, height, width)
    w_ptr,          # pointer to weight tensor (in_channels, 1, kernel_size, 1)
    out_ptr,        # pointer to output tensor
    batch_size,
    in_channels,
    out_height,
    out_width,
    height,
    width,
    kernel_size,
    stride,
    padding,
    dilation,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_c,
    weight_stride_kh,
    weight_stride_kw,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CH: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute program ids
    pid_b = tl.program_id(0)
    pid_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute offsets for blocks
    batch_start = pid_b * BLOCK_SIZE_BATCH
    ch_start = pid_ch * BLOCK_SIZE_CH
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # Define offsets within blocks
    b_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    ch_offsets = ch_start + tl.arange(0, BLOCK_SIZE_CH)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)

    # Load input and weights (broadcasting over batch and channel blocks)
    mask_b = b_offsets < batch_size
    mask_ch = ch_offsets < in_channels
    mask_h = h_offsets < out_height
    mask_w = w_offsets < out_width

    # Broadcast masks
    mask = mask_b[:, None, None, None] & mask_ch[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]

    # Output coordinates
    out_h = h_offsets[:, None] * stride - padding
    out_w = w_offsets[None, :] * stride - padding

    # Input base pointer
    input_ptrs = x_ptr + (
        b_offsets[:, None, None, None] * input_stride_b +
        ch_offsets[None, :, None, None] * input_stride_c +
        (out_h + 0)[:, None, :, None] * input_stride_h +
        (out_w + 0)[None, :, None, :] * input_stride_w
    )
    weight_ptrs = w_ptr + (
        ch_offsets[None, :, None, None] * weight_stride_c +
        tl.arange(0, kernel_size)[:, None, None, None] * weight_stride_kh +
        tl.zeros((kernel_size, 1, 1, 1), dtype=tl.int32) * weight_stride_kw
    )

    # Initialize output
    output = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_CH, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Convolution loop over kernel
    for ki in range(0, kernel_size):
        # Compute input position with dilation
        in_h = out_h + ki * dilation
        valid_h = (in_h >= 0) & (in_h < height)

        # Broadcast valid mask across width
        mask_hw = valid_h & ((out_w >= 0) & (out_w < width))[None, :, None, :]

        # Full mask including spatial validity
        load_mask = mask & mask_hw

        # Load input and weight
        x_vals = tl.load(input_ptrs + ki * dilation * input_stride_h, mask=load_mask, other=0.0)
        w_vals = tl.load(weight_ptrs + ki * weight_stride_kh, mask=mask_ch[None, :, None, None], other=0.0)

        # Multiply-accumulate
        output += x_vals * w_vals

    # Write output
    output_ptrs = out_ptr + (
        b_offsets[:, None, None, None] * output_stride_b +
        ch_offsets[None, :, None, None] * output_stride_c +
        h_offsets[:, None, None, None] * output_stride_h +
        w_offsets[None, None, None, :] * output_stride_w
    )
    tl.store(output_ptrs, output, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Asymmetric kernel: (kernel_size, 1)
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='linear')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        # Compute output dimensions
        out_height = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (width + 2 * self.padding - 1) // self.stride + 1

        # Allocate output
        out = torch.empty(batch_size, self.in_channels, out_height, out_width, device=x.device, dtype=x.dtype)

        # Get strides
        input_strides = x.stride()
        output_strides = out.stride()
        weight_strides = self.weight.stride()

        # Launch kernel
        def grid(META):
            return (
                triton.cdiv(batch_size, META['BLOCK_SIZE_BATCH']),
                triton.cdiv(self.in_channels, META['BLOCK_SIZE_CH']),
                triton.cdiv(out_height, META['BLOCK_SIZE_H']),
                triton.cdiv(out_width, META['BLOCK_SIZE_W']),
            )

        # Autotune block sizes
        depthwise_conv2d_kernel[grid](
            x, self.weight, out,
            batch_size, self.in_channels, out_height, out_width,
            height, width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            output_strides[0], output_strides[1], output_strides[2], output_strides[3],
            input_strides[0], input_strides[1], input_strides[2], input_strides[3],
            weight_strides[0], weight_strides[1], weight_strides[2],
            BLOCK_SIZE_BATCH=4,
            BLOCK_SIZE_CH=4,
            BLOCK_SIZE_H=16,
            BLOCK_SIZE_W=32,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out