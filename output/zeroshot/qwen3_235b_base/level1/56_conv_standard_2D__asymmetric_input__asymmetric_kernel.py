import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    out_height, out_width,
    groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)  # batch
    pid_g = tl.program_id(axis=1)  # group
    pid_out_h = tl.program_id(axis=2)  # output height
    pid_out_w = tl.program_id(axis=3)  # output width

    # Compute group parameters
    channels_per_group = in_channels // groups
    group_offset_w = pid_g * channels_per_group * kernel_h * kernel_w
    group_offset_in = pid_g * channels_per_group

    # Pointers to input and output
    x_ptr = x_ptr + pid_b * in_channels * height * width
    out_ptr = out_ptr + pid_b * out_channels * out_height * out_width + (pid_g * (out_channels // groups) + tl.arange(0, BLOCK_SIZE_M))[:, None] * out_height * out_width + (pid_out_h * out_width + pid_out_w) * 1

    # Offset for output channel within group
    out_ch_start = pid_g * (out_channels // groups)
    out_ch_range = out_ch_start + tl.arange(0, BLOCK_SIZE_M)

    # Input spatial start
    in_h_start = pid_out_h * stride_h - padding_h
    in_w_start = pid_out_w * stride_w - padding_w

    # Load input tiles
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for c in range(0, channels_per_group, BLOCK_SIZE_C):
        c_block = min(channels_per_group - c, BLOCK_SIZE_C)
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                h_offset = in_h_start + kh * dilation_h
                w_offset = in_w_start + kw * dilation_w
                mask_x = (
                    (h_offset >= 0) & (h_offset < height) &
                    (w_offset >= 0) & (w_offset < width) &
                    (tl.arange(0, BLOCK_SIZE_N) < 1)
                )
                offsets_x = (group_offset_in + c + tl.arange(0, BLOCK_SIZE_C)) * height * width + h_offset * width + w_offset
                x = tl.load(x_ptr + offsets_x, mask=mask_x[None, :] & (tl.arange(0, BLOCK_SIZE_C) < c_block)[None, :], other=0.0)

                # Load weights
                w_offset = group_offset_w + (c + tl.arange(0, BLOCK_SIZE_C)) * kernel_h * kernel_w + kh * kernel_w + kw
                w = tl.load(w_ptr + w_offset[:, None] + out_ch_range[None, :], mask=(tl.arange(0, BLOCK_SIZE_C) < c_block)[:, None], other=0.0)

                # Accumulate
                acc += tl.dot(w, x.to(tl.float32), out_dtype=tl.float32)

    # Store output
    mask_out = (out_ch_range < out_channels)[:, None] & (tl.arange(0, BLOCK_SIZE_N) < 1)
    tl.store(out_ptr, acc, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias:
            nn.init.zeros_(self.bias_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dimensions
        batch_size, in_channels, height, width = x.shape
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups

        # Output dimensions
        out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        # Output tensor
        out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

        # Define block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_C = 16
        BLOCK_SIZE_K = 32

        # Grid configuration
        grid = (
            batch_size,
            groups,
            out_height,
            out_width
        )

        # Launch kernel
        conv2d_kernel[grid](
            x, self.weight, out,
            batch_size, in_channels, out_channels, height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            out_height, out_width,
            groups,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        # Add bias if needed
        if self.bias:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out