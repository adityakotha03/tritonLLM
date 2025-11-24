import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, height, width, depth,
    out_channels, out_height, out_width, out_depth,
    kernel_h, kernel_w, kernel_d,
    stride_h, stride_w, stride_d,
    padding_h, padding_w, padding_d,
    dilation_h, dilation_w, dilation_d,
    groups,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w, input_stride_d,
    weight_stride_k, weight_stride_cg, weight_stride_h, weight_stride_w, weight_stride_d,
    output_stride_b, output_stride_k, output_stride_h, output_stride_w, output_stride_d,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_IN_CH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Compute program ids
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2) // out_depth
    pid_d = tl.program_id(axis=2) % out_depth

    # Handle grouped convolution
    group_id = pid_k // (out_channels // groups)
    ch_per_group = in_channels // groups
    weight_offset_c = group_id * ch_per_group

    # Compute output spatial indices
    h_out = pid_hw // out_width
    w_out = pid_hw % out_width
    d_out = pid_d

    # Bounds check
    if h_out >= out_height or w_out >= out_width or d_out >= out_depth:
        return

    # Compute input start position
    h_in = h_out * stride_h - padding_h
    w_in = w_out * stride_w - padding_w
    d_in = d_out * stride_d - padding_d

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH), dtype=tl.float32)

    # Iterate over input channels and kernel positions
    for ih in range(0, kernel_h):
        for iw in range(0, kernel_w):
            for id in range(0, kernel_d):
                h_offset = h_in + ih * dilation_h
                w_offset = w_in + iw * dilation_w
                d_offset = d_in + id * dilation_d

                # Check bounds
                mask_input = (
                    (h_offset >= 0) & (h_offset < height) &
                    (w_offset >= 0) & (w_offset < width) &
                    (d_offset >= 0) & (d_offset < depth)
                )

                # Load input tile: [BLOCK_SIZE_BATCH, BLOCK_SIZE_IN_CH]
                offs_b = tl.arange(0, BLOCK_SIZE_BATCH)
                offs_c = tl.arange(0, BLOCK_SIZE_IN_CH)
                input_mask = (
                    (offs_b[:, None] < batch) &
                    (offs_c[None, :] < ch_per_group) &
                    mask_input
                )
                input_offset = (
                    offs_b[:, None] * input_stride_b +
                    (weight_offset_c + offs_c[None, :]) * input_stride_c +
                    h_offset * input_stride_h +
                    w_offset * input_stride_w +
                    d_offset * input_stride_d
                )
                input_val = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)

                # Load weights: [BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH]
                offs_k = tl.arange(0, BLOCK_SIZE_OUT_CH)
                weight_mask = (
                    (offs_k[:, None] < BLOCK_SIZE_OUT_CH) &
                    (offs_c[None, :] < BLOCK_SIZE_IN_CH)
                )
                weight_offset = (
                    (pid_k + offs_k[:, None]) * weight_stride_k +
                    offs_c[None, :] * weight_stride_cg +
                    ih * weight_stride_h +
                    iw * weight_stride_w +
                    id * weight_stride_d
                )
                weight_val = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)

                # Accumulate: acc[b, k] += sum_c input[b, c] * weight[k, c]
                acc += tl.dot(input_val, weight_val, out_dtype=tl.float32)

    # Store output
    offs_b = tl.arange(0, BLOCK_SIZE_BATCH)
    offs_k = tl.arange(0, BLOCK_SIZE_OUT_CH)
    output_mask = (offs_b[:, None] < batch) & (offs_k[None, :] < BLOCK_SIZE_OUT_CH)
    output_offset = (
        offs_b[:, None] * output_stride_b +
        (pid_k + offs_k[None, :]) * output_stride_k +
        h_out * output_stride_h +
        w_out * output_stride_w +
        d_out * output_stride_d
    )
    output_val = acc.to(tl.float16)
    tl.store(output_ptr + output_offset, output_val, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and optional bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, 1)
        )
        if bias:
            self.bias_tensor = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias_tensor', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.bias_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width, depth = x.shape
        kernel_h, kernel_w, kernel_d = self.kernel_size, self.kernel_size, 1
        out_height = (height + 2 * self.padding - self.dilation * (kernel_h - 1) - 1) // self.stride + 1
        out_width = (width + 2 * self.padding - self.dilation * (kernel_w - 1) - 1) // self.stride + 1
        out_depth = (depth + 2 * self.padding - self.dilation * (kernel_d - 1) - 1) // self.stride + 1

        # Output shape
        out = torch.empty(batch, self.out_channels, out_height, out_width, out_depth, dtype=torch.float16, device=x.device)

        # Ensure contiguous inputs
        x = x.contiguous()
        weight = self.weight.contiguous()

        # Define block sizes
        BLOCK_SIZE_BATCH = triton.next_power_of_2(batch)
        BLOCK_SIZE_BATCH = min(BLOCK_SIZE_BATCH, 16)
        BLOCK_SIZE_OUT_CH = 16
        BLOCK_SIZE_IN_CH = 16
        BLOCK_SIZE_HW = 32
        BLOCK_SIZE_D = 8

        # Grid configuration
        num_hw_blocks = (out_height * out_width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
        num_d_blocks = out_depth
        grid = (
            (batch + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH,
            (self.out_channels + BLOCK_SIZE_OUT_CH - 1) // BLOCK_SIZE_OUT_CH,
            num_hw_blocks * num_d_blocks
        )

        # Launch kernel
        conv3d_kernel[grid](
            x, weight, out,
            batch, self.in_channels, height, width, depth,
            self.out_channels, out_height, out_width, out_depth,
            kernel_h, kernel_w, kernel_d,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            self.dilation, self.dilation, self.dilation,
            self.groups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_OUT_CH=BLOCK_SIZE_OUT_CH,
            BLOCK_SIZE_IN_CH=BLOCK_SIZE_IN_CH,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Add bias if needed
        if self.bias:
            out = out + self.bias_tensor.view(1, -1, 1, 1, 1)

        return out