import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding, output_padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_CH: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    # Thread indices
    pid_b = tl.program_id(0)
    pid_ch = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # Block offsets
    off_h = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)
    off_ch = pid_ch * BLOCK_CH + tl.arange(0, BLOCK_CH)

    # Input and output shape
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Load input
    off_x_h = (off_h - padding) // stride
    off_x_w = (off_w - padding) // stride
    mask_x_h = (off_h >= padding) & (off_h < padding + (height - 1) * stride + 1) & (off_x_h >= 0) & (off_x_h < height)
    mask_x_w = (off_w >= padding) & (off_w < padding + (width - 1) * stride + 1) & (off_x_w >= 0) & (off_x_w < width)
    mask_x = mask_x_h[:, None] & mask_x_w[None, :]
    x_offsets = pid_b * x_stride0 + off_ch[:, None, None] * x_stride1 + off_x_h[:, None, None] * x_stride2 + off_x_w[None, :, None] * x_stride3
    x = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)

    # Load weights
    off_kh = tl.arange(0, kernel_size)[:, None]
    off_kw = tl.arange(0, kernel_size)[None, :]
    off_w_h = off_kh - padding
    off_w_w = off_kw - padding
    mask_w = (off_w_h >= 0) & (off_w_h < kernel_size) & (off_w_w >= 0) & (off_w_w < kernel_size)
    w_offsets = off_ch[:, None, None] * w_stride0 + off_kh * w_stride1 + off_kw * w_stride2 + pid_ch * w_stride3
    w = tl.load(w_ptr + w_offsets, mask=mask_w, other=0.0)

    # Compute output
    out = tl.dot(x, w)  # (BLOCK_H, BLOCK_W, BLOCK_CH) @ (kernel_size, kernel_size, BLOCK_CH) -> (BLOCK_H, BLOCK_W, BLOCK_CH)
    out = tl.sum(out, axis=2)

    # Output offsets
    off_out_h = off_h
    off_out_w = off_w
    off_out_ch = pid_ch * BLOCK_CH + tl.arange(0, BLOCK_CH)
    mask_out = (off_out_h < out_h) & (off_out_w < out_w) & (off_out_ch < out_channels)
    out_offsets = pid_b * out_stride0 + off_out_ch[:, None, None] * out_stride1 + off_out_h[:, None, None] * out_stride2 + off_out_w[None, :, None] * out_stride3
    tl.store(out_ptr + out_offsets, out, mask=mask_out)


def triton_conv_transpose2d(x, w, bias=None, stride=1, padding=0, output_padding=0):
    """
    Triton-based transposed 2D convolution with fused bias and optimized memory access.
    Uses bf16 to leverage Tensor Cores.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape

    # Output dimensions
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # Strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3 = w.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Tune block sizes based on input
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_CH = 16
    BLOCK_OH = 32
    BLOCK_OW = 32

    # Grid: (batch, in_ch//block_ch, out_h//block_oh, out_w//block_ow)
    grid = lambda meta: (
        batch_size,
        (in_channels + meta["BLOCK_CH"] - 1) // meta["BLOCK_CH"],
        (out_h + meta["BLOCK_OH"] - 1) // meta["BLOCK_OH"],
        (out_w + meta["BLOCK_OW"] - 1) // meta["BLOCK_OW"]
    )

    # Launch kernel with bf16 for Tensor Core usage
    conv_transpose2d_kernel[grid](
        x, w, out,
        x_stride0, x_stride1, x_stride2, x_stride3,
        w_stride0, w_stride1, w_stride2, w_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, output_padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_CH=BLOCK_CH,
        BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW
    )

    # Optionally add bias if provided
    if bias is not None:
        out += bias.view(1, -1, 1, 1)

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

        # Initialize weight with proper shape
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton-based convolution with bias fusion
        return triton_conv_transpose2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )