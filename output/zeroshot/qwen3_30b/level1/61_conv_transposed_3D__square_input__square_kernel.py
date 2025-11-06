import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth, 
    height,
    width,
    out_depth,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_SIZE_D: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    # Thread indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Compute batch, output channel, and spatial indices
    batch = pid_b
    out_ch = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Compute input indices via convolution mapping
    in_d = out_d * stride - padding
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Check bounds
    out_ch_mask = out_ch < out_channels
    out_d_mask = out_d < out_depth
    out_h_mask = out_h < out_height
    out_w_mask = out_w < out_width
    out_mask = out_ch_mask[:, None, None, None] & out_d_mask[None, :, None, None] & out_h_mask[None, None, :, None] & out_w_mask[None, None, None, :]

    in_ch = tl.arange(0, in_channels)
    in_d = in_d[None, None, None, :]  # (1, 1, 1, BLOCK_SIZE_W)
    in_h = in_h[None, None, :, None]  # (1, 1, BLOCK_SIZE_H, 1)
    in_w = in_w[None, :, None, None]  # (1, BLOCK_SIZE_H, 1, 1)
    in_ch = in_ch[:, None, None, None]  # (in_channels, 1, 1, 1)

    # Reshape kernel: (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
    # Map to block indices
    w_offset = (pid_c // (out_channels // groups)) * in_channels * kernel_size * kernel_size * kernel_size + \
               (pid_c % (out_channels // groups)) * kernel_size * kernel_size * kernel_size + \
               in_ch * kernel_size * kernel_size + \
               (in_d // stride) * kernel_size + \
               (in_h // stride) * kernel_size + \
               (in_w // stride)

    # Handle output_padding
    out_pad_d = output_padding // 2 if output_padding > 0 else 0
    out_pad_h = output_padding // 2 if output_padding > 0 else 0
    out_pad_w = output_padding // 2 if output_padding > 0 else 0

    # Adjust indices if padded
    in_d = in_d + out_pad_d
    in_h = in_h + out_pad_h
    in_w = in_w + out_pad_w

    # Check input bounds
    in_d_mask = (in_d >= 0) & (in_d < depth)
    in_h_mask = (in_h >= 0) & (in_h < height)
    in_w_mask = (in_w >= 0) & (in_w < width)

    # Final mask combining output and input bounds
    mask = out_mask & in_d_mask & in_h_mask & in_w_mask

    # Load input tensor (x) in blocked fashion
    x = tl.load(
        x_ptr + batch * in_channels * depth * height * width +
        in_ch * depth * height * width +
        in_d * height * width +
        in_h * width +
        in_w,
        mask=mask,
        other=0.0
    )

    # Load weights (w)
    w = tl.load(
        w_ptr + pid_c * in_channels * kernel_size * kernel_size * kernel_size +
        in_ch * kernel_size * kernel_size * kernel_size +
        (in_d % kernel_size) * kernel_size * kernel_size +
        (in_h % kernel_size) * kernel_size +
        (in_w % kernel_size),
        mask=mask,
        other=0.0
    )

    # Compute output
    out = tl.dot(x, w)

    # Store output
    tl.store(
        out_ptr + batch * out_channels * out_depth * out_height * out_width +
        out_ch * out_depth * out_height * out_width +
        out_d * out_height * out_width +
        out_h * out_width +
        out_w,
        out,
        mask=mask
    )


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, stride: int, padding: int, output_padding: int, groups: int) -> torch.Tensor:
    # Ensure inputs are on GPU and contiguous
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Shape of input and output
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = w.shape
    out_depth = (depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding

    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)

    # Block sizes and tile sizes
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    TILE_SIZE_D = 32
    TILE_SIZE_H = 32
    TILE_SIZE_W = 32

    # Grid: (batch_size, ceil(out_channels / BLOCK_SIZE_C), ceil(out_depth / BLOCK_SIZE_D), ceil(out_height / BLOCK_SIZE_H), ceil(out_width / BLOCK_SIZE_W))
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        (out_depth + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (out_height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"]
    )

    # Launch kernel
    conv_transpose3d_kernel[grid](
        x,
        w,
        out,
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        TILE_SIZE_D=TILE_SIZE_D,
        TILE_SIZE_H=TILE_SIZE_H,
        TILE_SIZE_W=TILE_SIZE_W,
    )

    # Add bias if present
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups)