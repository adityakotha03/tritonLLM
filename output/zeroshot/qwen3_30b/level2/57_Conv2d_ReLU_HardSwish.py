import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    w_stride_0, w_stride_1, w_stride_2, w_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, pad, stride,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr
):
    # Get thread and block indices
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    pid_out_c = tl.program_id(3)

    # Calculate output coordinates
    out_h = pid_out_h * TILE_H + tl.arange(0, TILE_H)
    out_w = pid_out_w * TILE_W + tl.arange(0, TILE_W)
    out_c = pid_out_c * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    # Calculate valid output bounds
    valid_out_h = (out_h < height) & (out_h >= 0)
    valid_out_w = (out_w < width) & (out_w >= 0)
    valid_out_c = (out_c < out_channels) & (out_c >= 0)

    # Broadcast out_h and out_w across channels
    out_h = tl.broadcast_to(out_h[:, None], (TILE_H, BLOCK_SIZE_H))
    out_w = tl.broadcast_to(out_w[:, None], (TILE_H, BLOCK_SIZE_H))
    out_c = tl.broadcast_to(out_c[None, :], (TILE_H, BLOCK_SIZE_H))

    # Calculate input indices
    in_h = out_h * stride - pad
    in_w = out_w * stride - pad

    # Kernel indices
    k_h = tl.arange(0, kernel_size)[:, None, None]
    k_w = tl.arange(0, kernel_size)[None, :, None]

    # Input channels
    in_c = tl.arange(0, in_channels)[None, None, :]

    # Compute input and weight offsets
    x_offset = (
        pid_batch * x_stride_0 +
        in_c * x_stride_1 +
        in_h * x_stride_2 +
        in_w * x_stride_3
    )
    w_offset = (
        out_c * w_stride_0 +
        in_c * w_stride_1 +
        k_h * w_stride_2 +
        k_w * w_stride_3
    )

    # Load input and weight data
    x_data = tl.load(
        x_ptr + x_offset,
        mask=(valid_out_h[:, None, None] & valid_out_w[:, None, None] & (in_c[None, None, :] < in_channels)),
        other=0.0
    )
    w_data = tl.load(
        w_ptr + w_offset,
        mask=(valid_out_c[None, :, None] & (in_c[None, None, :] < in_channels) & (k_h < kernel_size) & (k_w < kernel_size)),
        other=0.0
    )

    # Compute output
    # Reduce over kernel and input channels
    out = tl.sum(x_data * w_data, axis=(3, 4))

    # Apply ReLU
    out = tl.maximum(out, 0.0)

    # Apply HardSwish: x * relu(x + 3) / 6
    x_plus_3 = out + 3.0
    relu_x_plus_3 = tl.maximum(x_plus_3, 0.0)
    hardswish = out * (relu_x_plus_3 / 6.0)
    hardswish = tl.clamp(hardswish, 0.0, 1.0)

    # Store result
    out_offset = (
        pid_batch * out_stride_0 +
        out_c * out_stride_1 +
        out_h * out_stride_2 +
        out_w * out_stride_3
    )
    tl.store(
        out_ptr + out_offset,
        hardswish,
        mask=valid_out_h[:, None, None] & valid_out_w[:, None, None] & valid_out_c[None, :, None]
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, stride=stride)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = stride

    def forward(self, x):
        # Ensure contiguous input
        x = x.contiguous()

        # Get shape info
        batch_size, _, height, width = x.shape
        out_channels, in_channels, kernel_h, kernel_w = self.conv.weight.shape

        # Prepare output
        out = torch.empty_like(x)

        # Use Triton kernel with appropriate block sizes
        # Use shared memory for weight tiling and reduce memory traffic
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        TILE_H = 16
        TILE_W = 16

        # Calculate grid
        num_blocks_h = (height + TILE_H - 1) // TILE_H
        num_blocks_w = (width + TILE_W - 1) // TILE_W
        num_blocks_c = (out_channels + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H

        grid = (batch_size, num_blocks_h, num_blocks_w, num_blocks_c)

        # Launch kernel
        conv_relu_hardswish_kernel[
            grid
        ](
            x,
            self.conv.weight,
            out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv.weight.stride(0), self.conv.weight.stride(1), self.conv.weight.stride(2), self.conv.weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch_size, in_channels, out_channels, height, width,
            kernel_size, self.pad, self.stride,
            BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
            TILE_H=TILE_H, TILE_W=TILE_W
        )

        return out

# Define the inputs
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]