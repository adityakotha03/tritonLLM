import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_h,
    stride_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Calculate block indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Calculate offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    c_offset = pid_c * BLOCK_SIZE_C

    # Load input and weights
    # Input: (batch_size, in_channels, height, width)
    # Weights: (out_channels, in_channels, 1, 1)
    # Output: (batch_size, out_channels, height, width)

    # Load input tile
    x_offsets = (
        pid_batch * in_channels * height * width +
        c_offset * height * width +
        h_offset * width +
        w_offset
    )
    x_mask = (
        (h_offset + tl.arange(0, BLOCK_SIZE_H)) < height
    ) & (
        (w_offset + tl.arange(0, BLOCK_SIZE_W)) < width
    )

    x = tl.load(
        x_ptr + x_offsets,
        mask=x_mask[:, None],
        other=0.0
    )

    # Load weights tile
    w_offsets = pid_c * in_channels + tl.arange(0, BLOCK_SIZE_C)
    w_mask = w_offsets < in_channels
    w = tl.load(
        w_ptr + w_offsets,
        mask=w_mask[None, :],
        other=0.0
    )

    # Perform convolution: sum over in_channels
    # Weights are shared across spatial dimensions
    # x: [BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C]
    # w: [BLOCK_SIZE_C, in_channels]
    # Output: [BLOCK_SIZE_H, BLOCK_SIZE_W]
    out = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for i in range(0, in_channels, BLOCK_SIZE_C):
        w_part = tl.load(
            w_ptr + pid_c * in_channels + i,
            mask=(tl.arange(0, BLOCK_SIZE_C) + i) < in_channels,
            other=0.0
        )
        x_part = tl.load(
            x_ptr + x_offsets + i * height * width,
            mask=x_mask[:, None] & (tl.arange(0, BLOCK_SIZE_C)[None, :] + i < in_channels),
            other=0.0
        )
        out += tl.sum(x_part * w_part[None, :], axis=1)

    # Store output
    out_offsets = pid_batch * out_channels * height * width + \
                  pid_c * height * width + \
                  h_offset * width + \
                  w_offset
    out_mask = x_mask
    tl.store(out_ptr + out_offsets, out, mask=out_mask)


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None):
    """
    Optimized 1x1 Conv2D using Triton kernel with spatial and channel tiling.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = w.shape

    # Output tensor
    out = torch.empty(batch_size, out_channels, height, width, dtype=x.dtype, device=x.device)

    # Define block sizes (powers of 2)
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 16  # Must divide in_channels and out_channels

    # Calculate grid dimensions
    grid_h = triton.cdiv(height, BLOCK_SIZE_H)
    grid_w = triton.cdiv(width, BLOCK_SIZE_W)
    grid_c = triton.cdiv(out_channels, BLOCK_SIZE_C)

    # Grid: (batch, h, w, c)
    grid = (batch_size, grid_h, grid_w, grid_c)

    # Launch kernel
    conv1d_kernel[grid](
        x,
        w,
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        1, 1,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    # Add bias if present
    if bias is not None:
        out += bias.view(1, out_channels, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        # We'll keep the original Conv2d for now, but use Triton to replace its forward pass
        # The weight will be copied to Triton for optimized execution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton kernel for 1x1 convolution
        return triton_conv1d(x, self.conv1d.weight, self.conv1d.bias if self.conv1d.bias is not None else None)