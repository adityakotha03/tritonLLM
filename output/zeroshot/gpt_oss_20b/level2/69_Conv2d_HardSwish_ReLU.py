import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: depthwise fused conv2d + HardSwish + ReLU
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def conv2d_hswish_relu_fused_kernel(
    X_ptr,
    W_ptr,
    out_ptr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    stride_dilation_h,
    stride_dilation_w,
    H,
    W,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Triton kernel that performs a 2D convolution with a square kernel,
    followed by HardSwish and ReLU activations, all fused together.
    The kernel works with BF16 input and weight tensors and outputs BF16.
    """
    # Grid indices
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)

    # Coordinates of the output tile
    row_start = block_row * BLOCK_H
    col_start = block_col * BLOCK_W

    # Allocate accumulator
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Loop over output channels
    for oc in range(out_channels):
        # Loop over kernel height
        for kh in range(kernel_h):
            # Loop over kernel width
            for kw in range(kernel_w):
                # Compute input coordinates
                in_row = row_start + tl.arange(0, BLOCK_H) * stride_h - pad_h + kh * dilation_h
                in_col = col_start + tl.arange(0, BLOCK_W) * stride_w - pad_w + kw * dilation_w

                # Broadcast in_row and in_col to tile shape
                in_row = tl.broadcast_to(in_row[:, None], [BLOCK_H, BLOCK_W])
                in_col = tl.broadcast_to(in_col[None, :], [BLOCK_H, BLOCK_W])

                # Mask to avoid out of bounds
                mask = (in_row >= 0) & (in_row < H) & (in_col >= 0) & (in_col < W)

                # Load input values
                # X_ptr is (B, C, H, W). Assume batch size 1 for simplicity.
                x_offsets = (
                    in_row * W + in_col
                )  # offset per element within a single channel
                x_offsets = x_offsets[None, :, :]  # shape [1, BLOCK_H, BLOCK_W]
                x_offsets = x_offsets + oc * H * W  # add channel offset
                x = tl.load(X_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

                # Load weight
                w_offset = oc * kernel_h * kernel_w + kh * kernel_w + kw
                w = tl.load(W_ptr + w_offset).to(tl.float32)

                acc += x * w

    # HardSwish: x * relu6(x + 3) / 6
    relu6 = tl.maximum(tl.minimum(acc + 3.0, 6.0), 0.0)
    hswish = acc * relu6 / 6.0

    # ReLU
    out = tl.maximum(hswish, 0.0)

    # Store result
    out_offsets = (
        tl.arange(0, BLOCK_H)[:, None] * W + tl.arange(0, BLOCK_W)[None, :]
    )
    out_ptr = out_ptr + out_offsets
    tl.store(out_ptr, out.to(tl.bfloat16), mask=mask)


def conv2d_hswish_relu_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
):
    """
    Wrapper for the fused convolution + HardSwish + ReLU.
    Parameters:
        x: Input tensor of shape (B, C_in, H, W) in BF16.
        weight: Weight tensor of shape (C_out, C_in, kH, kW) in BF16.
        stride: Tuple of (stride_h, stride_w).
        padding: Tuple of (pad_h, pad_w).
        dilation: Tuple of (dilation_h, dilation_w).
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out) in BF16.
    """
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    # Compute output dimensions
    H_out = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    # Flatten batch dimension for kernel launch (assume B==1 for simplicity)
    # For B>1, launch multiple grid rows
    grid = lambda meta: (
        (H_out + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W_out + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    conv2d_hswish_relu_fused_kernel[grid](
        x, weight, out,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        stride[0] * dilation[0], stride[1] * dilation[1],
        H, W,
        C_in, C_out,
        kH, kW,
        BLOCK_H=16,
        BLOCK_W=16,
    )
    return out


# ----------------------------------------------------------------------
# ModelNew definition
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that uses a fused Triton kernel for Conv2d + HardSwish + ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16, device="cuda")
        )
        self.stride = (1, 1)
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.dilation = (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is BF16 and contiguous
        x = x.to(torch.bfloat16).contiguous()
        out = conv2d_hswish_relu_fused(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return out