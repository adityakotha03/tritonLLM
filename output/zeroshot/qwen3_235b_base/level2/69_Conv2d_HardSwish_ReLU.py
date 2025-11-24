import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv2d_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, out_channels, out_height, out_width,
    in_channels, in_height, in_width,
    kernel_size_h, kernel_size_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c_out, weight_stride_c_in, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_IN_CH: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 2D convolution implemented with implicit gemm (image to column + matmul)
    # Then fused activation: hardswish(x) = x * relu6(x + 3) / 6, then relu
    # So final: relu( conv(x) * relu6(conv(x) + 3) / 6 )

    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)

    # Compute offsets for output tiles
    b_start = pid_b * BLOCK_SIZE_BATCH
    c_start = pid_c * BLOCK_SIZE_OUT_CH
    hw_start = pid_hw * BLOCK_SIZE_HW

    # Define offsets for the current block
    b_range = b_start + tl.arange(0, BLOCK_SIZE_BATCH)
    c_range = c_start + tl.arange(0, BLOCK_SIZE_OUT_CH)
    hw_range = hw_start + tl.arange(0, BLOCK_SIZE_HW)

    # Masks to avoid out of bounds
    b_mask = b_range < batch_size
    c_mask = c_range < out_channels
    hw_mask = hw_range < out_height * out_width

    # Flat output indices
    out_h = (hw_range // out_width)
    out_w = (hw_range % out_width)

    # Input spatial locations for convolution
    in_h_base = out_h * stride_h - padding_h
    in_w_base = out_w * stride_w - padding_w
    in_h_k = in_h_base[:, None] + dilation_h * tl.arange(0, kernel_size_h)[None, :]
    in_w_k = in_w_base[:, None] + dilation_w * tl.arange(0, kernel_size_w)[None, :]

    # Load bias (per output channel)
    bias = tl.load(
        b_ptr + c_range,
        mask=c_mask, other=0.0
    )  # (BLOCK_SIZE_OUT_CH,)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_HW), dtype=tl.float32)

    # Loop over input channels in tiles
    in_ch_block_count = triton.cdiv(in_channels, BLOCK_SIZE_IN_CH)
    for ic in range(0, in_ch_block_count):
        ic_start = ic * BLOCK_SIZE_IN_CH
        ic_range = ic_start + tl.arange(0, BLOCK_SIZE_IN_CH)
        ic_mask = ic_range < in_channels

        # Load input block: (BLOCK_SIZE_BATCH, BLOCK_SIZE_IN_CH, BLOCK_SIZE_HW, KERNEL_H, KERNEL_W)
        # We will use a loop over kernel for simplicity and memory efficiency
        in_ptrs = x_ptr + \
                  (b_range[:, None, None, None, None] * input_stride_b) + \
                  (ic_range[None, :, None, None, None] * input_stride_c) + \
                  (in_h_k[None, None, :, :, None] * input_stride_h) + \
                  (in_w_k[None, None, :, None, :] * input_stride_w)
        in_masks = b_mask[:, None, None, None, None] & \
                   ic_mask[None, :, None, None, None] & \
                   (in_h_k[None, None, :, :, None] >= 0) & \
                   (in_h_k[None, None, :, :, None] < in_height) & \
                   (in_w_k[None, None, :, None, :] >= 0) & \
                   (in_w_k[None, None, :, None, :] < in_width)

        input_val = tl.load(in_ptrs, mask=in_masks, other=0.0)  # (B, IC, HW, KH, KW)

        # Reshape input for matmul: (B * HW, IC * KH * KW)
        input_flat = tl.reshape(input_val, (BLOCK_SIZE_BATCH * BLOCK_SIZE_HW, BLOCK_SIZE_IN_CH * kernel_size_h * kernel_size_w))

        # Load weights: (OC, IC, KH, KW) -> (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH * KH * KW)
        w_ptrs = w_ptr + \
                 (c_range[:, None] * weight_stride_c_out) + \
                 (ic_range[None, :] * weight_stride_c_in) + \
                 (tl.arange(0, kernel_size_h)[None, :] * weight_stride_h) + \
                 (tl.arange(0, kernel_size_w)[:, None] * weight_stride_w)
        w_masks = c_mask[:, None] & ic_mask[None, :]  # (OC, IC)
        w_tiles = tl.load(w_ptrs, mask=w_masks, other=0.0)  # (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH, KH, KW)
        w_flat = tl.reshape(w_tiles, (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH * kernel_size_h * kernel_size_w))

        # Perform matmul: acc[B, OC, HW] += w[OC, IC*KH*KW] @ input[B*HW, IC*KH*KW].T
        # So we do: (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_BATCH * BLOCK_SIZE_HW) = (BLOCK_SIZE_OUT_CH, K) @ (K, B*HW)
        acc += tl.dot(w_flat, tl.trans(input_flat))

    # Add bias: (BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_HW)
    acc = acc + bias[:, None]

    # Apply Hardswish: x * relu6(x + 3) / 6
    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    hardswish = acc * relu6 / 6.0

    # Then apply ReLU
    out = tl.maximum(hardswish, 0.0)

    # Store output
    out_ptrs = out_ptr + \
               (b_range[:, None, None] * output_stride_b) + \
               (c_range[None, :, None] * output_stride_c) + \
               (out_h[None, None, :] * output_stride_h) + \
               (out_w[None, None, :] * output_stride_w)
    out_masks = b_mask[:, None, None] & c_mask[None, :, None] & hw_mask[None, None, :]
    tl.store(out_ptrs, out, mask=out_masks)


def triton_conv2d_hardswish_relu(x, weight, bias, stride, padding, dilation):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape

    # Compute output spatial dimensions
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_size_h - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_size_w - 1) - 1) // stride[1] + 1

    # Output tensor
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_SIZE_BATCH = 2
    BLOCK_SIZE_OUT_CH = 16
    BLOCK_SIZE_HW = 32
    BLOCK_SIZE_IN_CH = 8
    BLOCK_SIZE_K = 32  # Not used directly, but for autotuning

    # Grid
    grid = (
        triton.cdiv(batch_size, BLOCK_SIZE_BATCH),
        triton.cdiv(out_channels, BLOCK_SIZE_OUT_CH),
        triton.cdiv(out_height * out_width, BLOCK_SIZE_HW)
    )

    # Launch kernel
    _conv2d_hardswish_relu_kernel[grid](
        x, weight, bias, out,
        batch_size, out_channels, out_height, out_width,
        in_channels, in_height, in_width,
        kernel_size_h, kernel_size_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_OUT_CH=BLOCK_SIZE_OUT_CH,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_IN_CH=BLOCK_SIZE_IN_CH,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused convolution, HardSwish, and ReLU using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Register weight and bias as parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        # Init
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return triton_conv2d_hardswish_relu(
            x, self.weight, self.bias,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1)
        )