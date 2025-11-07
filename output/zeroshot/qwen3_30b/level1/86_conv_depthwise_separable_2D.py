import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, out_ptr,
    H, W, C, KH, KW, stride, padding,
    n_elements,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Block offsets
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W
    c_offset = pid_c * BLOCK_C

    # Compute the output coordinates
    out_h = h_offset // stride
    out_w = w_offset // stride

    # Check if the output is within bounds
    if out_h >= H or out_w >= W:
        return

    # Compute input indices with padding
    h_indices = h_offset + tl.arange(0, BLOCK_H)
    w_indices = w_offset + tl.arange(0, BLOCK_W)

    # Mask for out-of-bounds input indices (after padding)
    h_mask = (h_indices >= 0) & (h_indices < H + 2 * padding)
    w_mask = (w_indices >= 0) & (w_indices < W + 2 * padding)

    # Adjust input indices to account for padding
    h_idx = h_indices - padding
    w_idx = w_indices - padding

    # Create masks for valid input coordinates
    valid_h = h_mask & (h_idx >= 0) & (h_idx < H)
    valid_w = w_mask & (w_idx >= 0) & (w_idx < W)

    # Combine masks for valid input locations
    valid = valid_h[:, None] & valid_w[None, :]

    # Load input data for this block (only valid positions)
    x = tl.load(
        x_ptr + (tl.broadcast_to(h_idx[:, None], (BLOCK_H, BLOCK_W)) * W + tl.broadcast_to(w_idx[None, :], (BLOCK_H, BLOCK_W))) * C,
        mask=valid[:, :, None],
        other=0.0
    )
    x = tl.load(
        x_ptr + (tl.broadcast_to(h_idx[:, None], (BLOCK_H, BLOCK_W)) * W + tl.broadcast_to(w_idx[None, :], (BLOCK_H, BLOCK_W))) * C,
        mask=valid[:, :, None],
        other=0.0
    )

    # Reshape x to (BLOCK_H * BLOCK_W, C)
    x = x.view(BLOCK_H * BLOCK_W, C)

    # Load kernel weights (KH x KW x C)
    w = tl.load(
        w_ptr + (tl.arange(0, KH)[:, None] * KW + tl.arange(0, KW)[None, :]) * C,
        mask=tl.arange(0, KH)[:, None] < KH,
        other=0.0
    )
    w = w.view(KH * KW, C)

    # Perform convolution (depthwise): output = x * w, sum over kernel spatial dims
    out = tl.dot(x, w, allow_tf32=True)

    # Apply stride and store output
    out_h = h_offset // stride
    out_w = w_offset // stride

    # Only store if within output bounds
    out_valid = (out_h < H // stride) & (out_w < W // stride)
    if out_valid:
        tl.store(
            out_ptr + (out_h * (W // stride) + out_w) * C + tl.arange(0, C),
            out,
            mask=tl.arange(0, C) < C
        )


@triton.jit
def pointwise_conv_kernel(
    x_ptr, w_ptr, out_ptr,
    H, W, C_in, C_out,
    n_elements,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W
    c_offset = pid_c * BLOCK_C

    # Check output bounds
    if h_offset >= H or w_offset >= W:
        return

    # Load input (C_in channels)
    x = tl.load(
        x_ptr + (h_offset * W + w_offset) * C_in + tl.arange(0, C_in),
        mask=tl.arange(0, C_in) < C_in,
        other=0.0
    )

    # Load weights (C_in x C_out)
    w = tl.load(
        w_ptr + tl.arange(0, C_in)[:, None] * C_out + tl.arange(0, C_out)[None, :],
        mask=tl.arange(0, C_in)[:, None] < C_in,
        other=0.0
    )

    # Matrix multiply: output = x @ w
    out = tl.dot(x, w, allow_tf32=True)

    # Store output
    tl.store(
        out_ptr + (h_offset * W + w_offset) * C_out + tl.arange(0, C_out),
        out,
        mask=tl.arange(0, C_out) < C_out
    )


def triton_depthwise_conv(x, w, stride, padding):
    batch_size, in_channels, H, W = x.shape
    KH, KW = w.shape[2], w.shape[3]

    # Output dimensions
    out_H = (H + 2 * padding - KH) // stride + 1
    out_W = (W + 2 * padding - KW) // stride + 1

    # Output tensor
    out = torch.empty(batch_size, in_channels, out_H, out_W, device=x.device, dtype=x.dtype)

    # Configure kernel launch
    BLOCK_H, BLOCK_W = 16, 16
    BLOCK_C = 32

    # Grid: (out_H / BLOCK_H, out_W / BLOCK_W, in_channels / BLOCK_C)
    grid_h = triton.cdiv(out_H, BLOCK_H)
    grid_w = triton.cdiv(out_W, BLOCK_W)
    grid_c = triton.cdiv(in_channels, BLOCK_C)

    # Launch kernel
    depthwise_conv_kernel[
        (grid_h, grid_w, grid_c),
    ](
        x, w, out,
        H, W, in_channels, KH, KW, stride, padding,
        out.numel(),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )

    return out


def triton_pointwise_conv(x, w):
    batch_size, in_channels, H, W = x.shape
    out_channels = w.shape[0]

    # Output tensor
    out = torch.empty(batch_size, out_channels, H, W, device=x.device, dtype=x.dtype)

    # Configure kernel launch
    BLOCK_H, BLOCK_W = 16, 16
    BLOCK_C = 32

    grid_h = triton.cdiv(H, BLOCK_H)
    grid_w = triton.cdiv(W, BLOCK_W)
    grid_c = triton.cdiv(in_channels, BLOCK_C)

    # Launch kernel
    pointwise_conv_kernel[
        (grid_h, grid_w, grid_c),
    ](
        x, w, out,
        H, W, in_channels, out_channels,
        out.numel(),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize depthwise and pointwise kernels
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()

        # Apply depthwise convolution via Triton
        x = triton_depthwise_conv(x, self.depthwise_weight, self.stride, self.padding)

        # Apply pointwise convolution via Triton
        x = triton_pointwise_conv(x, self.pointwise_weight)

        # Add bias if present
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)

        return x