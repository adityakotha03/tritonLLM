import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, out_channels, out_depth, out_height, out_width,
    in_channels, in_depth, in_height, in_width,
    kernel_size, stride, padding, groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_g, weight_stride_r, weight_stride_s, weight_stride_t,
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute starting positions
    batch_start = pid_b * BLOCK_SIZE_BATCH
    channel_start = pid_c * BLOCK_SIZE_OUT_CHANNEL
    hw_start = pid_hw * BLOCK_SIZE_HW

    # Offsets within blocks
    b_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    c_offsets = channel_start + tl.arange(0, BLOCK_SIZE_OUT_CHANNEL)
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)

    # Mask for valid batches and channels
    b_mask = b_offsets < batch_size
    c_mask = c_offsets < out_channels
    hw_mask = hw_offsets < out_height * out_width

    # Broadcast masks
    mask = b_mask[:, None, None] & c_mask[None, :, None] & hw_mask[None, None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNEL, BLOCK_SIZE_HW), dtype=tl.float32)

    # Group parameters
    channels_per_group = in_channels // groups
    out_c_per_group = out_channels // groups
    group_id = c_offsets // out_c_per_group
    local_c = c_offsets % out_c_per_group

    # Iterate over input channels within group
    for ic in range(0, channels_per_group):
        group_base = group_id * channels_per_group + ic
        weight_group_offset = group_id * weight_stride_g

        # Load input: shape (BLOCK_SIZE_BATCH, channels_per_group, in_depth, in_height, in_width)
        for kd in range(0, kernel_size):
            for kh in range(0, kernel_size):
                for kw in range(0, kernel_size):
                    # Output indices
                    do = tl.arange(0, BLOCK_SIZE_BATCH)[:, None, None]  # batch
                    co = c_offsets[None, :, None]  # output channel
                    ho = hw_offsets[None, None, :] // out_width  # output height
                    wo = hw_offsets[None, None, :] % out_width  # output width
                    # Compute input indices
                    di = (do * stride - padding + kd).to(tl.int32)
                    hi = (ho * stride - padding + kh).to(tl.int32)
                    wi = (wo * stride - padding + kw).to(tl.int32)
                    # Validity mask
                    valid = (di >= 0) & (di < in_depth) & (hi >= 0) & (hi < in_height) & (wi >= 0) & (wi < in_width)
                    valid_mask = valid & mask
                    # Input pointer offsets
                    input_offsets = (
                        do * input_stride_b + group_base * input_stride_c +
                        di * input_stride_d + hi * input_stride_h + wi * input_stride_w
                    )
                    input_vals = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
                    # Weight pointer offsets
                    weight_offsets = (
                        weight_group_offset +
                        local_c * weight_stride_k +
                        kd * weight_stride_r + kh * weight_stride_s + kw * weight_stride_t
                    )
                    weight_vals = tl.load(weight_ptr + weight_offsets, mask=c_mask, other=0.0)
                    # Multiply and accumulate
                    acc += input_vals * weight_vals[None, :, None]

    # Store output
    output_offsets = (
        b_offsets[:, None, None] * output_stride_b +
        c_offsets[None, :, None] * output_stride_c +
        (hw_offsets[None, None, :] // out_width) * output_stride_d +
        (hw_offsets[None, None, :] // out_width) * output_stride_h +
        (hw_offsets[None, None, :] % out_width) * output_stride_w
    )
    tl.store(output_ptr + output_offsets, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.output_padding = output_padding
        self.bias = bias

        # Initialize weight and optional bias
        k = 1.0 / (in_channels * kernel_size ** 3)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.weight.data.uniform_(-k**0.5, k**0.5)
        if bias:
            self.bias_val = nn.Parameter(torch.empty(out_channels))
            self.bias_val.data.uniform_(-k**0.5, k**0.5)
        else:
            self.bias_val = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dimensions
        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        device = x.device
        dtype = x.dtype

        # Output dimensions
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Output tensor
        out = torch.zeros(batch_size, self.out_channels, out_depth, out_height, out_width, dtype=dtype, device=device)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
                triton.cdiv(self.out_channels, meta['BLOCK_SIZE_OUT_CHANNEL']),
                triton.cdiv(out_height * out_width, meta['BLOCK_SIZE_HW'])
            )

        # Autotune block sizes
        BLOCK_SIZE_BATCH = 4
        BLOCK_SIZE_OUT_CHANNEL = 16
        BLOCK_SIZE_HW = 64

        conv_transpose3d_kernel[grid](
            x, self.weight, out,
            batch_size, self.out_channels, out_depth, out_height, out_width,
            in_channels, in_depth, in_height, in_width,
            self.kernel_size, self.stride, self.padding, self.groups,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_OUT_CHANNEL=BLOCK_SIZE_OUT_CHANNEL,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        )

        # Add bias if needed
        if self.bias_val is not None:
            out += self.bias_val.view(1, -1, 1, 1, 1)

        return out