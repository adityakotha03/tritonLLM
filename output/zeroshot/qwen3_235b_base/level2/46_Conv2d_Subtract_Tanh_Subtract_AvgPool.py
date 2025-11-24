import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_tanh_sub_sub_avgpool_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    stride_x, stride_y, stride_h, stride_w,
    stride_out_x, stride_out_y, stride_out_h, stride_out_w,
    stride_weight_r, stride_weight_c, stride_weight_in, stride_weight_out,
    stride_bias,
    height: tl.constexpr, width: tl.constexpr,
    in_channels: tl.constexpr, out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    pool_kernel: tl.constexpr,
    subtract1_value: tl.constexpr, subtract2_value: tl.constexpr,
    output_height: tl.constexpr, output_width: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)
    pid_oc = tl.program_id(3)

    # Calculate input patch start
    h_offset = pid_oh * pool_kernel
    w_offset = pid_ow * pool_kernel

    # Define offsets for blocks
    c_offsets = pid_oc * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    h_offsets = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_offset + tl.arange(0, BLOCK_SIZE_W)

    # Load input tiles
    input_mask = (
        (c_offsets[:, None, None, None] < in_channels) &
        (h_offsets[None, :, None, None] >= padding) &
        (h_offsets[None, :, None, None] < height + padding) &
        (w_offsets[None, None, :, None] < width + padding) &
        (w_offsets[None, None, :, None] >= padding)
    )
    h_padded = h_offsets - padding
    w_padded = w_offsets - padding
    x_ptrs = x_ptr + (
        pid_b * stride_x +
        c_offsets[:, None, None, None] * stride_y +
        h_padded[None, :, None, None] * stride_h +
        w_padded[None, None, :, None] * stride_w
    )
    x = tl.load(x_ptrs, mask=input_mask, other=0.0)

    # Convolution: load weights and compute
    weight_mask = (
        (c_offsets[:, None, None, None] < in_channels) &
        (k_offsets[None, :, None, None] < kernel_size) &
        (k_offsets[None, None, :, None] < kernel_size) &
        (k_offsets[None, None, None, :] < out_channels)
    )
    weight_ptrs = weight_ptr + (
        k_offsets[None, :, None, None] * stride_weight_r +
        k_offsets[None, None, :, None] * stride_weight_c +
        c_offsets[:, None, None, None] * stride_weight_in +
        k_offsets[None, None, None, :] * stride_weight_out
    )
    weights = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

    # Compute convolution
    conv_out = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for ki in range(kernel_size):
        for kj in range(kernel_size):
            h_grid = h_padded + padding - ki
            w_grid = w_padded + padding - kj
            mask_inside = (
                (h_grid[None, :, None, None] >= 0) &
                (h_grid[None, :, None, None] < height) &
                (w_grid[None, None, :, None] >= 0) &
                (w_grid[None, None, :, None] < width)
            )
            x_selected = tl.where(mask_inside, x, 0.0)
            w_selected = weights[:, ki, kj, :]
            conv_out += tl.einsum('chwk,cok->howk', x_selected, w_selected)

    # Add bias
    bias_ptrs = bias_ptr + k_offsets * stride_bias
    bias_mask = k_offsets < out_channels
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    conv_out += bias[None, None, None, :]

    # Subtract1, tanh, subtract2
    conv_out = conv_out - subtract1_value
    conv_out = tl.tanh(conv_out)
    conv_out = conv_out - subtract2_value

    # Average pooling over pool_kernel x pool_kernel
    pool_mask = (
        (h_offsets[None, :, None] >= h_offset) &
        (h_offsets[None, :, None] < h_offset + pool_kernel) &
        (w_offsets[None, None, :] >= w_offset) &
        (w_offsets[None, None, :] < w_offset + pool_kernel)
    )
    pooled = tl.sum(conv_out * pool_mask[:, :, :, None], axis=(1, 2)) / (pool_kernel * pool_kernel)

    # Store output
    out_h = pid_oh
    out_w = pid_ow
    out_c_start = pid_oc * BLOCK_SIZE_C
    c_mask = c_offsets < out_channels
    out_ptrs = out_ptr + (
        pid_b * stride_out_x +
        c_offsets * stride_out_y +
        out_h * stride_out_h +
        out_w * stride_out_w
    )
    tl.store(out_ptrs, pooled, mask=c_mask[:, None])


class ModelNew(nn.Module):
    """
    Optimized model with fused Triton kernel for conv2d, subtract, tanh, subtract, and avgpool.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size

    def forward(self, x):
        # Get output dimensions
        batch_size, _, height, width = x.shape
        out_height = height // self.kernel_size_pool
        out_width = width // self.kernel_size_pool

        # Output tensor
        out = torch.empty(batch_size, self.conv.out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

        # Launch kernel
        def grid(meta):
            return (
                batch_size,
                triton.cdiv(out_height, meta['BLOCK_SIZE_H']),
                triton.cdiv(out_width, meta['BLOCK_SIZE_W']),
                triton.cdiv(self.conv.out_channels, meta['BLOCK_SIZE_C']),
            )

        # Use autotuning
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 4, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 4, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 4, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
            ],
            key=['out_channels'],
        )
        @triton.jit
        def _kernel_caller(
            x_ptr, weight_ptr, bias_ptr, out_ptr,
            stride_x, stride_y, stride_h, stride_w,
            stride_out_x, stride_out_y, stride_out_h, stride_out_w,
            stride_weight_r, stride_weight_c, stride_weight_in, stride_weight_out,
            stride_bias,
            height, width,
            in_channels, out_channels,
            kernel_size,
            pool_kernel,
            subtract1_value, subtract2_value,
            output_height, output_width,
            padding,
            BLOCK_SIZE_C: tl.constexpr,
            BLOCK_SIZE_H: tl.constexpr,
            BLOCK_SIZE_W: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
        ):
            conv2d_tanh_sub_sub_avgpool_kernel(
                x_ptr, weight_ptr, bias_ptr, out_ptr,
                stride_x, stride_y, stride_h, stride_w,
                stride_out_x, stride_out_y, stride_out_h, stride_out_w,
                stride_weight_r, stride_weight_c, stride_weight_in, stride_weight_out,
                stride_bias,
                height, width,
                in_channels, out_channels,
                kernel_size,
                pool_kernel,
                subtract1_value, subtract2_value,
                output_height, output_width,
                padding,
                BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_K
            )

        _kernel_caller[grid](
            x, self.conv.weight, self.conv.bias, out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            self.conv.weight.stride(0), self.conv.weight.stride(1), self.conv.weight.stride(2), self.conv.weight.stride(3),
            self.conv.bias.stride(0),
            height, width,
            self.conv.in_channels, self.conv.out_channels,
            self.kernel_size,
            self.kernel_size_pool,
            self.subtract1_value, self.subtract2_value,
            out_height, out_width,
            self.padding
        )

        return out