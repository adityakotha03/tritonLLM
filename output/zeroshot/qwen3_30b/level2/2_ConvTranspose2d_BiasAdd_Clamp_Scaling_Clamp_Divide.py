import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scalar_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    out_channels: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program instance processes a spatial block of size (BLOCK_SIZE_H, BLOCK_SIZE_W)
    block_h = tl.program_id(0)
    block_w = tl.program_id(1)

    out_h_start = block_h * BLOCK_SIZE_H
    out_w_start = block_w * BLOCK_SIZE_W

    # Loop over batch
    for batch in range(batch_size):
        # Loop over output channels
        for c_out in range(out_channels):
            # Loop over spatial dimensions in the block
            for h in range(BLOCK_SIZE_H):
                out_h = out_h_start + h
                if out_h >= H_out:
                    continue
                for w in range(BLOCK_SIZE_W):
                    out_w = out_w_start + w
                    if out_w >= W_out:
                        continue

                    # Load the input value
                    x_val = tl.load(
                        x_ptr + (batch * out_channels + c_out) * H_out * W_out + out_h * W_out + out_w,
                        mask=(out_h < H_out) & (out_w < W_out),
                        other=0.0
                    )
                    # Load the bias
                    bias_val = tl.load(
                        bias_ptr + c_out,
                        mask=(out_h < H_out) & (out_w < W_out),
                        other=0.0
                    )
                    # Add bias
                    z = x_val + bias_val
                    # Clamp to [0, 1]
                    z = tl.clamp(z, 0.0, 1.0)
                    # Scale and clamp
                    # Final: min(clamp(z, 0, 1), 1.0 / scaling_factor)
                    z = tl.min(z, 1.0 / scaling_factor)
                    # Store
                    tl.store(
                        output_ptr + (batch * out_channels + c_out) * H_out * W_out + out_h * W_out + out_w,
                        z,
                        mask=(out_h < H_out) & (out_w < W_out)
                    )


def triton_scalar_operations(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)

    # Parameters
    batch_size = x.shape[0]
    out_channels = x.shape[1]
    H_out = x.shape[2]
    W_out = x.shape[3]

    # Tune block size
    # We'll use BLOCK_SIZE_H and BLOCK_SIZE_W of 16 for now
    # We can autotune
    # For now, we'll use a fixed block size.
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Number of spatial blocks
    num_blocks_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W

    # Grid
    grid = lambda meta: (meta['num_blocks_h'], meta['num_blocks_w'])

    # Launch the kernel
    scalar_kernel[grid](
        x, bias, out,
        batch_size=batch_size,
        out_channels=out_channels,
        H_out