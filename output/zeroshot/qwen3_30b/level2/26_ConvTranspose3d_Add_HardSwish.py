import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    x_ptr,  # Pointer to input tensor (batch, in_channels, D, H, W)
    w_ptr,  # Pointer to weight tensor (out_channels, in_channels, kD, kH, kW)
    bias_ptr,  # Pointer to bias tensor (out_channels, 1, 1, 1, 1)
    out_ptr,  # Pointer to output tensor (batch, out_channels, D_out, H_out, W_out)
    batch_size, in_channels, out_channels,
    D, H, W, D_out, H_out, W_out,
    kD, kH, kW,
    stride, padding, output_padding,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    TILE_SIZE_D: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    # Define thread indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Calculate block offsets
    block_b_start = pid_b * BLOCK_SIZE_B
    block_c_start = pid_c * BLOCK_SIZE_C
    block_d_start = pid_d * BLOCK_SIZE_D
    block_h_start = pid_h * BLOCK_SIZE_H
    block_w_start = pid_w * BLOCK_SIZE_W

    # Calculate tile offsets
    tile_d = tl.arange(0, TILE_SIZE_D)[:, None, None]
    tile_h = tl.arange(0, TILE_SIZE_H)[None, :, None]
    tile_w = tl.arange(0, TILE_SIZE_W)[None, None, :]

    # Get input dimensions
    in_d_offset = block_d_start - padding
    in_h_offset = block_h_start - padding
    in_w_offset = block_w_start - padding

    # Compute output indices
    out_d = block_d_start + tile_d
    out_h = block_h_start + tile_h
    out_w = block_w_start + tile_w

    # Create mask for output indices
    out_d_mask = (out_d < D_out)
    out_h_mask = (out_h < H_out)
    out_w_mask = (out_w < W_out)
    out_mask = out_d_mask & out_h_mask & out_w_mask

    # Get input indices for the current block
    in_d = in_d_offset + tile_d * stride
    in_h = in_h_offset + tile_h * stride
    in_w = in_w_offset + tile_w * stride

    # Clamp input indices to valid range
    in_d = tl.clip(in_d, 0, D - 1)
    in_h = tl.clip(in_h, 0, H - 1)
    in_w = tl.clip(in_w, 0, W - 1)

    # Load input tensor for current block
    x_block = tl.load(
        x_ptr + (pid_b * in_channels * D * H * W +
                 pid_c * D * H * W +
                 in_d * H * W + in_h * W + in_w),
        mask=tl.broadcast(out_mask, (TILE_SIZE_D, TILE_SIZE_H, TILE_SIZE_W)),
        other=0.0
    )

    # Load weights for current output channel
    w_block = tl.load(
        w_ptr + (pid_c * in_channels * kD * kH * kW +
                 tl.arange(0, in_channels)[:, None, None, None] *
                 kD * kH * kW +
                 tl.arange(0, kD)[:, None, None] * kH * kW +
                 tl.arange(0, kH)[:, None] * kW +
                 tl.arange(0, kW)),
        mask=tl.broadcast(out_mask, (in_channels, kD, kH, kW)),
        other=0.0
    )

    # Perform convolution transpose via sliding window
    # Inner product over in_channels and spatial dims
    acc = tl.zeros((TILE_SIZE_D, TILE_SIZE_H, TILE_SIZE_W), dtype=tl.float32)
    for c in range(in_channels):
        for d in range(kD):
            for h in range(kH):
                for w in range(kW):
                    in_val = x_block[c, d, h, w]
                    w_val = w_block[c, d, h, w]
                    acc += in_val * w_val

    # Apply bias and handle output padding
    bias = tl.load(bias_ptr + pid_c, mask=tl.arange(0, out_channels) == pid_c)
    acc = acc + bias

    # Store output with proper mask
    tl.store(
        out_ptr + (pid_b * out_channels * D_out * H_out * W_out +
                   pid_c * D_out * H_out * W_out +
                   out_d * H_out * W_out + out_h * W_out + out_w),
        acc,
        mask=out_mask
    )


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # HardSwish: x * (x + 3) / 6
    out = x * (x + 3.0) * (1.0 / 6.0)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose_3d(x, w, bias, stride, padding, output_padding):
    """
    Custom Triton-based 3D transposed convolution with bias.
    """
    assert x.is_cuda and w.is_cuda and bias.is_cuda, "All inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kD, kH, kW = w.shape
    D_out = (D - 1) * stride + kD - 2 * padding + output_padding
    H_out = (H - 1) * stride + kH - 2 * padding + output_padding
    W_out = (W - 1) * stride + kW - 2 * padding + output_padding

    out = torch.empty(batch_size, out_channels, D_out, H_out, W_out, device=x.device, dtype=x.dtype)

    # Define kernel parameters
    BLOCK_SIZE_B = 128
    BLOCK_SIZE_C = 128
    BLOCK_SIZE_D = 32
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    TILE_SIZE_D = 16
    TILE_SIZE_H = 16
    TILE_SIZE_W = 16

    # Grid dimensions
    grid = (
        (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B,
        (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,
        (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )

    # Launch kernel
    conv_transpose_3d_kernel[grid](
        x, w, bias, out,
        batch_size, in_channels, out_channels,
        D, H, W, D_out, H_out, W_out,
        kD, kH, kW,
        stride, padding, output_padding,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        TILE_SIZE_D=TILE_SIZE_D,
        TILE_SIZE_H=TILE_SIZE_H,
        TILE_SIZE_W=TILE_SIZE_W
    )

    return out


def triton_hardswish(x):
    """
    Triton-based HardSwish activation: x * (x + 3) / 6
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        # Use custom Triton kernel for conv_transpose
        x = triton_conv_transpose_3d(x, self.conv_transpose.weight, self.bias, self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0])
        
        # Add input
        x = x + add_input
        
        # Apply HardSwish using Triton kernel
        x = triton_hardswish(x)
        
        return x