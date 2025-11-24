import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, y_ptr,
    batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out,
    kernel_size, stride, padding, output_padding, groups,
    stride_xh, stride_xw, stride_xc,
    stride_wh, stride_ww, stride_wc,
    stride_yh, stride_yw, stride_yc,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_IN_CH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Compute program ids
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Calculate offsets for output spatial dimensions
    oh_start = pid_hw * BLOCK_SIZE_HW
    ow_start = 0
    oh_range = tl.arange(0, BLOCK_SIZE_HW)
    ow_range = tl.arange(0, BLOCK_SIZE_HW)

    # Bounds for output height and width
    oh_mask = oh_start + oh_range < height_out
    ow_mask = ow_start + ow_range < width_out
    oh = tl.where(oh_mask, oh_start + oh_range, 0)
    ow = tl.where(ow_mask, ow_start + ow_range, 0)

    # Loop over input channels per group
    group_id = pid_out_ch // (out_channels // groups)
    in_ch_per_group = in_channels // groups
    out_ch_per_group = out_channels // groups
    in_ch_base = group_id * in_ch_per_group
    out_ch_base = group_id * out_ch_per_group

    # Initialize output accumulator
    acc = tl.zeros([BLOCK_SIZE_HW, BLOCK_SIZE_HW], dtype=tl.float32)

    # Loop over input channels in blocks
    for in_ch_block in range(0, in_ch_per_group, BLOCK_SIZE_IN_CH):
        in_ch_offset = in_ch_base + in_ch_block + tl.arange(0, BLOCK_SIZE_IN_CH)
        in_ch_mask = in_ch_offset < in_channels
        in_ch_offset = tl.where(in_ch_mask, in_ch_offset, 0)

        # Compute input spatial coordinates
        ih = (oh - output_padding) // stride
        iw = (ow - output_padding) // stride

        # Check if input coordinates are valid
        ih_valid = (ih >= padding) & (ih < height_in + padding)
        iw_valid = (iw >= padding) & (iw < width_in + padding)
        valid = ih_valid[:, None] & iw_valid[None, :]  # [BLOCK_SIZE_HW, BLOCK_SIZE_HW]

        # Load input tiles: [BLOCK_SIZE_HW, BLOCK_SIZE_HW, BLOCK_SIZE_IN_CH]
        x_offsets = (
            pid_batch * stride_xc * in_channels + in_ch_offset[None, None, :] * stride_xc +
            (ih - padding)[:, None, None] * stride_xh +
            (iw - padding)[None, :, None] * stride_xw
        )
        x_mask = valid[:, :, None] & in_ch_mask[None, None, :]
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # Load weights: [BLOCK_SIZE_IN_CH, kernel_size, kernel_size]
        for out_ch_rel in range(BLOCK_SIZE_OUT_CH):
            out_ch = pid_out_ch * BLOCK_SIZE_OUT_CH + out_ch_rel
            if out_ch >= out_channels:
                continue
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    w_offset = (
                        out_ch * stride_wc +
                        (kh - kernel_size // 2 + padding) * stride_wh +
                        (kw - kernel_size // 2 + padding) * stride_ww
                    )
                    w = tl.load(w_ptr + w_offset)
                    # Accumulate: x * w
                    acc += x[:, :, out_ch_rel] * w

        # Store output
        y_offsets = (
            pid_batch * stride_yc * out_channels +
            pid_out_ch * stride_yc +
            oh[:, None] * stride_yh +
            ow[None, :] * stride_yw
        )
        y_mask = oh_mask[:, None] & ow_mask[None, :]
        tl.store(y_ptr + y_offsets, acc.to(tl.float16), mask=y_mask)


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
        self.bias = bias

        # Initialize weight tensor (out_ch, in_ch, k, k)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias_tensor = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias_tensor', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if bias:
            nn.init.zeros_(self.bias_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()

        # Compute output shape
        height_out = (x.shape[2] - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        width_out = (x.shape[3] - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Create output tensor
        y = torch.empty(x.shape[0], self.out_channels, height_out, width_out, dtype=torch.float16, device=x.device)

        # Get strides
        stride_xh, stride_xw, stride_xc = x.stride(2), x.stride(3), x.stride(1)
        stride_wh, stride_ww, stride_wc = self.weight.stride(2), self.weight.stride(3), self.weight.stride(0)
        stride_yh, stride_yw, stride_yc = y.stride(2), y.stride(3), y.stride(1)

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(x.shape[0], meta['BLOCK_SIZE_BATCH']),
                triton.cdiv(self.out_channels, meta['BLOCK_SIZE_OUT_CH']),
                triton.cdiv(height_out, meta['BLOCK_SIZE_HW'])
            )

        # Autotune block sizes
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CH': 16, 'BLOCK_SIZE_IN_CH': 16, 'BLOCK_SIZE_HW': 16}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CH': 32, 'BLOCK_SIZE_IN_CH': 16, 'BLOCK_SIZE_HW': 16}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CH': 16, 'BLOCK_SIZE_IN_CH': 32, 'BLOCK_SIZE_HW': 16}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CH': 32, 'BLOCK_SIZE_IN_CH': 32, 'BLOCK_SIZE_HW': 16}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CH': 64, 'BLOCK_SIZE_IN_CH': 32, 'BLOCK_SIZE_HW': 16}, num_stages=3, num_warps=4),
            ],
            key=['in_channels', 'out_channels', 'height_out', 'width_out']
        )
        @triton.jit
        def _kernel(
            x_ptr, w_ptr, y_ptr,
            batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out,
            kernel_size, stride, padding, output_padding, groups,
            stride_xh, stride_xw, stride_xc,
            stride_wh, stride_ww, stride_wc,
            stride_yh, stride_yw, stride_yc,
            BLOCK_SIZE_BATCH: tl.constexpr,
            BLOCK_SIZE_OUT_CH: tl.constexpr,
            BLOCK_SIZE_IN_CH: tl.constexpr,
            BLOCK_SIZE_HW: tl.constexpr,
        ):
            conv_transpose2d_kernel(
                x_ptr, w_ptr, y_ptr,
                batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out,
                kernel_size, stride, padding, output_padding, groups,
                stride_xh, stride_xw, stride_xc,
                stride_wh, stride_ww, stride_wc,
                stride_yh, stride_yw, stride_yc,
                BLOCK_SIZE_BATCH,
                BLOCK_SIZE_OUT_CH,
                BLOCK_SIZE_IN_CH,
                BLOCK_SIZE_HW,
            )

        _kernel[grid](
            x, self.weight, y,
            x.shape[0], self.in_channels, self.out_channels, x.shape[2], x.shape[3], height_out, width_out,
            self.kernel_size, self.stride, self.padding, self.output_padding, self.groups,
            stride_xh, stride_xw, stride_xc,
            stride_wh, stride_ww, stride_wc,
            stride_yh, stride_yw, stride_yc,
        )

        # Add bias if needed
        if self.bias:
            y = y + self.bias_tensor.view(1, -1, 1, 1)

        return y