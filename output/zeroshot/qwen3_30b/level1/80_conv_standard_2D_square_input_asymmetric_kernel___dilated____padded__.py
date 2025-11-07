import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor (optional)
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_channel_stride,
    weight_out_channel_stride,
    weight_kernel_h_stride,
    weight_kernel_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    batch_size,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    height,
    width,
    out_height,
    out_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
    TILE_SIZE_C: tl.constexpr,
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Block offsets
    block_start_h = pid_h * TILE_SIZE_H
    block_start_w = pid_w * TILE_SIZE_W
    block_start_c = pid_c * TILE_SIZE_C

    # Offsets within the block
    offs_h = tl.arange(0, TILE_SIZE_H)
    offs_w = tl.arange(0, TILE_SIZE_W)
    offs_c = tl.arange(0, TILE_SIZE_C)

    # Compute output coordinates
    out_h = block_start_h + offs_h
    out_w = block_start_w + offs_w
    out_c = block_start_c + offs_c

    # Mask out of bounds
    mask_h = out_h < out_height
    mask_w = out_w < out_width
    mask_c = out_c < out_channels
    mask = mask_h[:, None, None] & mask_w[None, :, None] & mask_c[None, None, :]

    # Output buffer
    acc = tl.zeros((TILE_SIZE_H, TILE_SIZE_W, TILE_SIZE_C), dtype=tl.float32)

    # Loop over input channels and kernel
    for c in range(0, in_channels, BLOCK_SIZE_C):
        # Input channel offset
        c_offset = c
        c_mask = c_offset + offs_c < in_channels

        # Load input block
        input_ptrs = input_ptr + \
            (pid_batch * input_batch_stride +
             c_offset * input_channel_stride +
             (out_h // stride_h - pad_h) * input_height_stride +
             (out_w // stride_w - pad_w) * input_width_stride)

        # Compute effective kernel indices
        kernel_h_idx = tl.arange(0, kernel_h) * dilation_h
        kernel_w_idx = tl.arange(0, kernel_w) * dilation_w

        # Load input values (with bounds checking)
        input_vals = tl.load(
            input_ptrs[:, :, None] + kernel_h_idx[None, None, :] * input_height_stride +
            kernel_w_idx[None, :, None] * input_width_stride,
            mask=(out_h[:, None, None] < out_height) &
                 (out_w[None, :, None] < out_width) &
                 (c_offset + offs_c[None, None, :] < in_channels),
            other=0.0
        )

        # Load weights for this block
        weight_ptrs = weight_ptr + \
            (block_start_c * weight_out_channel_stride +
             c_offset * weight_channel_stride +
             (kernel_h_idx[:, None] * weight_kernel_h_stride +
              kernel_w_idx[None, :] * weight_kernel_w_stride))

        weights = tl.load(
            weight_ptrs,
            mask=(c_offset + offs_c[None, None, :] < in_channels) &
                 (out_c[None, None, :] < out_channels),
            other=0.0
        )

        # Compute partial dot product
        acc += tl.dot(input_vals, weights)

    # Apply bias if exists
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_c, mask=mask_c)
        acc += bias[None, None, :]

    # Store output
    out_ptrs = output_ptr + \
        (pid_batch * output_batch_stride +
         block_start_c * output_channel_stride +
         block_start_h * output_height_stride +
         block_start_w * output_width_stride)
    tl.store(out_ptrs, acc, mask=mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1)):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # Compute output dimensions
    out_height = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride + 1
    out_width = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride + 1

    # Allocate output
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Get strides
    input_batch_stride = in_channels * height * width
    input_channel_stride = height * width
    input_height_stride = width
    input_width_stride = 1

    weight_out_channel_stride = in_channels * kernel_h * kernel_w
    weight_channel_stride = kernel_h * kernel_w
    weight_kernel_h_stride = kernel_w
    weight_kernel_w_stride = 1

    output_batch_stride = out_channels * out_height * out_width
    output_channel_stride = out_height * out_width
    output_height_stride = out_width
    output_width_stride = 1

    # Tune block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16
    TILE_SIZE_H = 16
    TILE_SIZE_W = 16
    TILE_SIZE_C = 16

    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["TILE_SIZE_C"] - 1) // meta["TILE_SIZE_C"],
        (out_height + meta["TILE_SIZE_H"] - 1) // meta["TILE_SIZE_H"],
        (out_width + meta["TILE_SIZE_W"] - 1) // meta["TILE_SIZE_W"]
    )

    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out, bias,
        input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
        weight_out_channel_stride, weight_channel_stride, weight_kernel_h_stride, weight_kernel_w_stride,
        output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
        batch_size, in_channels, out_channels, kernel_h, kernel_w, stride, stride, pad_h, pad_w,
        dilation_h, dilation_w, height, width, out_height, out_width,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C,
        TILE_SIZE_H=TILE_SIZE_H, TILE_SIZE_W=TILE_SIZE_W, TILE_SIZE_C=TILE_SIZE_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton-optimized convolution
        return triton_conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)