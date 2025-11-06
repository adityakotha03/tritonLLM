import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, 
    weight_ptr, 
    output_ptr,
    input_stride0, input_stride1, input_stride2, input_stride3,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3,
    output_stride0, output_stride1, output_stride2, output_stride3,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C: tl.constexpr,
):
    # Thread indexing
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output spatial indices
    h_start = pid_h * TILE_H - padding
    w_start = pid_w * TILE_W - padding

    # Compute output channel index
    c_start = pid_c * TILE_C

    # Compute output block dimensions
    h_end = h_start + TILE_H
    w_end = w_start + TILE_W

    # Create output offsets
    out_offsets_h = tl.arange(0, TILE_H)
    out_offsets_w = tl.arange(0, TILE_W)
    out_offsets_c = tl.arange(0, TILE_C)

    # Compute output indices
    out_h = h_start + out_offsets_h
    out_w = w_start + out_offsets_w
    out_c = c_start + out_offsets_c

    # Load output mask (valid within bounds)
    out_mask_h = (out_h >= 0) & (out_h < height)
    out_mask_w = (out_w >= 0) & (out_w < width)
    out_mask_c = (out_c < out_channels)

    out_mask = out_mask_h[:, None, None] & out_mask_w[None, :, None] & out_mask_c[None, None, :]

    # Compute input indices (after padding)
    in_h = out_h[None, :, None] * stride
    in_w = out_w[None, None, :] * stride
    in_c = tl.arange(0, in_channels)[:, None, None]

    # Compute kernel indices
    kh = tl.arange(0, kernel_size)[:, None, None]
    kw = tl.arange(0, kernel_size)[None, :, None]

    # Compute input and weight offsets
    input_offsets = (
        pid_batch * input_stride0 +
        in_c * input_stride1 +
        in_h * input_stride2 +
        in_w * input_stride3
    )
    weight_offsets = (
        out_c[:, None, None] * weight_stride0 +
        in_c[None, :, None] * weight_stride1 +
        kh[None, :, None] * weight_stride2 +
        kw[None, None, :] * weight_stride3
    )

    # Load input and weight tiles
    input_tile = tl.load(input_ptr + input_offsets, mask=(in_h < height) & (in_w < width), other=0.0)
    weight_tile = tl.load(weight_ptr + weight_offsets, mask=True)

    # Perform convolution: compute dot product across in_channels and kernel_size
    # Use tensor core accumulation
    accum = tl.zeros((TILE_H, TILE_W, TILE_C), dtype=tl.float32)
    for i in range(0, in_channels, BLOCK_C):
        i_end = min(i + BLOCK_C, in_channels)
        input_chunk = tl.load(
            input_ptr + input_offsets + i * input_stride1,
            mask=(in_h < height) & (in_w < width) & (in_c < i_end),
            other=0.0
        )
        weight_chunk = tl.load(
            weight_ptr + weight_offsets + i * weight_stride1,
            mask=(out_c < out_channels) & (in_c < i_end),
            other=0.0
        )
        # Compute partial dot product
        dot = tl.dot(input_chunk, weight_chunk)
        accum += dot

    # Store output with mask
    out_offsets = (
        pid_batch * output_stride0 +
        out_c[None, None, :] * output_stride1 +
        out_h[:, None, None] * output_stride2 +
        out_w[None, :, None] * output_stride3
    )
    tl.store(output_ptr + out_offsets, accum, mask=out_mask)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int = 4, padding: int = 2):
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Input shape: (B, C_in, H, W)
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Output dimensions
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, dtype=input.dtype, device=input.device)

    # Strides
    input_stride0, input_stride1, input_stride2, input_stride3 = input.stride()
    weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
    output_stride0, output_stride1, output_stride2, output_stride3 = output.stride()

    # Define block sizes
    BLOCK_H, BLOCK_W, BLOCK_C = 16, 16, 32
    TILE_H, TILE_W, TILE_C = 16, 16, 16

    # Grid: (batch, out_channels, out_h, out_w)
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["TILE_C"] - 1) // meta["TILE_C"],
        (out_h + meta["TILE_H"] - 1) // meta["TILE_H"],
        (out_w + meta["TILE_W"] - 1) // meta["TILE_W"]
    )

    # Launch kernel
    conv2d_kernel[
        grid
    ](
        input, weight, output,
        input_stride0, input_stride1, input_stride2, input_stride3,
        weight_stride0, weight_stride1, weight_stride2, weight_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
        TILE_H=TILE_H, TILE_W=TILE_W, TILE_C=TILE_C
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        # Replace torch.nn.functional.conv2d with custom Triton implementation
        return triton_conv2d(x, self.conv1.weight, stride=4, padding=2)