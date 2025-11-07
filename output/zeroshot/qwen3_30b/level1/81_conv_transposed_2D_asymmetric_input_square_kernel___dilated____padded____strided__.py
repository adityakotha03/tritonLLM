import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Pointer to input tensor (batch, in_channels, height_in, width_in)
    w_ptr,  # Pointer to weight tensor (out_channels, in_channels, kernel_h, kernel_w)
    out_ptr,  # Pointer to output tensor (batch, out_channels, height_out, width_out)
    batch_size, in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    kernel_size, stride, padding, dilation,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,  # Number of output channels per block
):
    # Shared memory for tiles
    # Use shared memory to cache parts of weights and inputs
    shared_x = tl.load(tl.make_block_ptr(x_ptr, (batch_size, in_channels, height_in, width_in), (in_channels * height_in * width_in, height_in * width_in, width_in, 1), (0, 0, 0, 0), (BLOCK_H, BLOCK_W, BLOCK_C, 1), (0, 0, 0, 0)))
    shared_w = tl.load(tl.make_block_ptr(w_ptr, (out_channels, in_channels, kernel_size, kernel_size), (in_channels * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1), (0, 0, 0, 0), (BLOCK_C, in_channels, kernel_size, kernel_size), (0, 0, 0, 0)))

    # Thread indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Calculate output position
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Output tile shape
    tile_h = tl.minimum(BLOCK_H, height_out - pid_h * BLOCK_H)
    tile_w = tl.minimum(BLOCK_W, width_out - pid_w * BLOCK_W)

    # Initialize output tile
    acc = tl.zeros((tile_h, tile_w, BLOCK_C), dtype=tl.float32)

    # Loop over input channels and kernel
    for k_c in range(0, in_channels, BLOCK_C):
        # Load input tile (block of input channels)
        start_c = k_c
        end_c = tl.minimum(k_c + BLOCK_C, in_channels)
        # Input: (batch_size, in_channels, height_in, width_in)
        # We only need one batch, one channel block
        x_block = tl.load(
            tl.make_block_ptr(x_ptr, (batch_size, in_channels, height_in, width_in), 
                              (in_channels * height_in * width_in, height_in * width_in, width_in, 1), 
                              (0, start_c, 0, 0), 
                              (1, end_c - start_c, BLOCK_H, BLOCK_W), 
                              (0, 0, 0, 0))
        )
        
        # Weight tile (for the current output channel block)
        w_block = tl.load(
            tl.make_block_ptr(w_ptr, (out_channels, in_channels, kernel_size, kernel_size), 
                              (in_channels * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1), 
                              (pid_c * BLOCK_C, start_c, 0, 0), 
                              (BLOCK_C, end_c - start_c, kernel_size, kernel_size), 
                              (0, 0, 0, 0))
        )

        # Pad input to handle padding
        # Use padding, dilation
        for h in range(kernel_size):
            for w in range(kernel_size):
                if dilation == 1:
                    h_pad = out_h * stride + h
                    w_pad = out_w * stride + w
                else:
                    h_pad = out_h * stride + h * dilation
                    w_pad = out_w * stride + w * dilation

                # Check bounds
                h_pad_mask = (h_pad >= 0) & (h_pad < height_in)
                w_pad_mask = (w_pad >= 0) & (w_pad < width_in)

                # Mask for valid input coordinates
                valid = h_pad_mask & w_pad_mask
                valid = valid[:, :, None]  # Expand for channel dimension

                # Load input values at (h_pad, w_pad) and apply mask
                x_val = tl.load(tl.make_block_ptr(x_ptr, (batch_size, in_channels, height_in, width_in),
                                                  (in_channels * height_in * width_in, height_in * width_in, width_in, 1),
                                                  (0, start_c, h_pad, w_pad),
                                                  (1, end_c - start_c, 1, 1),
                                                  (0, 0, 0, 0)),
                                mask=valid,
                                other=0.0)

                # Apply weight
                w_val = w_block[:, :, h, w]  # (BLOCK_C, end_c - start_c)

                # Accumulate
                acc += tl.expand_dims(x_val, -1) * tl.expand_dims(w_val, 0)  # Broadcasting

    # Store output
    # Only store valid output positions
    out_h_mask = out_h < height_out
    out_w_mask = out_w < width_out
    mask = out_h_mask[:, None] & out_w_mask[None, :]  # Shape: (tile_h, tile_w)

    # Store
    tl.store(
        tl.make_block_ptr(out_ptr, (batch_size, out_channels, height_out, width_out),
                          (out_channels * height_out * width_out, height_out * width_out, width_out, 1),
                          (pid_batch, pid_c * BLOCK_C, pid_h * BLOCK_H, pid_w * BLOCK_W),
                          (1, BLOCK_C, tile_h, tile_w),
                          (0, 0, 0, 0)),
        acc,
        mask=mask
    )


def triton_conv_transpose2d(x, w, stride, padding, dilation, out_channels, kernel_size, height_out, width_out):
    # Ensure inputs are on GPU and contiguous
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height_in, width_in = x.shape
    out = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)

    # Tunable parameters
    BLOCK_H = 64
    BLOCK_W = 64
    BLOCK_C = 16  # Must be divisible by block size in channel dimension

    # Grid setup: (batch, height_blocks, width_blocks, out_channel_blocks)
    num_blocks_h = (height_out + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (width_out + BLOCK_W - 1) // BLOCK_W
    num_blocks_c = (out_channels + BLOCK_C - 1) // BLOCK_C

    grid = lambda meta: (batch_size, num_blocks_h, num_blocks_w, num_blocks_c)

    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        height_in, width_in,
        height_out, width_out,
        kernel_size, stride, padding, dilation,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        batch_size, in_channels, height_in, width_in = x.shape
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

        # Call optimized Triton kernel
        out = triton_conv_transpose2d(
            x, self.weight, self.stride, self.padding, self.dilation,
            self.out_channels, self.kernel_size, height_out, width_out
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, self.out_channels, 1, 1)

        return out