import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, out_channels, input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_g, weight_stride_r, weight_stride_s, weight_stride_t,
    output_stride_b, output_stride_k, output_stride_p, output_stride_q, output_stride_r,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_IN_CHANNEL: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_pq = tl.program_id(2)

    # Compute output spatial indices (p, q, r)
    pid_p = pid_pq // output_width
    pid_r = pid_pq % output_width
    pid_q = pid_p % output_height
    pid_p = pid_p // output_height

    # Handle block-level tiling
    batch_start = pid_b * BLOCK_SIZE_BATCH
    k_start = pid_k * BLOCK_SIZE_OUT_CHANNEL

    # Pointers into input and output
    output_offs_b = tl.arange(0, BLOCK_SIZE_BATCH)
    output_offs_k = k_start + tl.arange(0, BLOCK_SIZE_OUT_CHANNEL)
    output_offs_p = pid_p * stride_d - padding_d
    output_offs_q = pid_q * stride_h - padding_h
    output_offs_r = pid_r * stride_w - padding_w

    # Input loading offsets
    input_d = output_offs_p[:, None, None, None] + dilation_d * tl.arange(0, kernel_d)[None, :, None, None]
    input_h = output_offs_q[:, None, None] + dilation_h * tl.arange(0, kernel_h)[None, :, None]
    input_w = output_offs_r[:, None] + dilation_w * tl.arange(0, kernel_w)[None, :]

    # Create masks for valid input accesses
    input_d_mask = (input_d >= 0) & (input_d < input_depth)
    input_h_mask = (input_h >= 0) & (input_h < input_height)
    input_w_mask = (input_w >= 0) & (input_w < input_width)
    input_mask_dh = input_d_mask & input_h_mask[:, None, :, :]
    input_mask = input_mask_dh & input_w_mask[:, None, None, :, :]

    # Broadcast input indices
    input_d = tl.where(input_d_mask, input_d, 0)
    input_h = tl.where(input_h_mask, input_h, 0)
    input_w = tl.where(input_w_mask, input_w, 0)

    # Input base offsets
    input_base_off = (
        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH)) * input_stride_b +
        input_d * input_stride_d +
        input_h * input_stride_h +
        input_w * input_stride_w
    )

    # Weight layout: (out_channels, in_channels_per_group, kernel_d, kernel_h, kernel_w)
    group = pid_k // (out_channels // groups)
    weight_k = output_offs_k - group * (out_channels // groups)
    weight_k = weight_k[:, None, None, None, None]
    weight_g = group
    weight_r = tl.arange(0, kernel_d)[None, :, None, None, None]
    weight_s = tl.arange(0, kernel_h)[None, None, :, None, None]
    weight_t = tl.arange(0, kernel_w)[None, None, None, :, None]

    weight_off = (
        weight_k * weight_stride_k +
        weight_g * weight_stride_g +
        weight_r * weight_stride_r +
        weight_s * weight_stride_s +
        weight_t * weight_stride_t
    )

    # Accumulate output
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNEL), dtype=tl.float32)

    # Iterate over input channels
    for c in range(0, in_channels // groups):
        input_off_c = input_base_off + (c + group * (in_channels // groups)) * input_stride_c
        input_vals = tl.load(input_ptr + input_off_c, mask=input_mask, other=0.0)
        weight_vals = tl.load(weight_ptr + weight_off + c * weight_stride_k, mask=None, other=0.0)
        # Contract over D, H, W
        input_vals = input_vals.to(tl.float32)
        weight_vals = weight_vals.to(tl.float32)
        inner_prod = tl.sum(input_vals * weight_vals, axis=[2, 3, 4])
        acc += inner_prod

    # Store output
    output_off_b = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    output_off_k = output_offs_k
    output_mask = (output_off_b[:, None] < batch_size) & (output_off_k[None, :] < out_channels)
    output_off = (
        output_off_b[:, None] * output_stride_b +
        output_off_k[None, :] * output_stride_k +
        pid_p * output_stride_p +
        pid_q * output_stride_q +
        pid_r * output_stride_r
    )
    tl.store(output_ptr + output_off, acc, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weight
        k_d, k_h, k_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, k_d, k_h, k_w))
        if bias:
            self.bias_tensor = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias_tensor', None)

        # Init using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if bias:
            nn.init.zeros_(self.bias_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get shapes
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        out_channels = self.out_channels
        k_d, k_h, k_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation
        groups = self.groups

        # Compute output spatial dimensions
        output_depth = (input_depth + 2 * pad_d - dilation_d * (k_d - 1) - 1) // stride_d + 1
        output_height = (input_height + 2 * pad_h - dilation_h * (k_h - 1) - 1) // stride_h + 1
        output_width = (input_width + 2 * pad_w - dilation_w * (k_w - 1) - 1) // stride_w + 1

        # Output tensor
        out = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
                triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT_CHANNEL']),
                output_depth * output_height * output_width
            )

        # Autotune block sizes
        conv3d_kernel[grid](
            x, self.weight, out,
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width,
            k_d, k_h, k_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w,
            groups,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
            BLOCK_SIZE_BATCH=4,
            BLOCK_SIZE_OUT_CHANNEL=16,
            BLOCK_SIZE_IN_CHANNEL=16,
            BLOCK_SIZE_D=16,
            BLOCK_SIZE_HW=32,
        )

        # Add bias if needed
        if self.bias:
            out = out + self.bias_tensor.view(1, -1, 1, 1, 1)

        return out