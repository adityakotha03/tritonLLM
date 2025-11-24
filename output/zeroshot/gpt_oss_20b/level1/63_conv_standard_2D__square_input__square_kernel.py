import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------
# Triton kernel for 2D convolution
# --------------------------------------------------
@triton.jit
def conv2d_kernel(
    input_ptr,          # [N, C, H, W]
    weight_ptr,         # [M, C, kH, kW]
    bias_ptr,           # [M]  (optional)
    output_ptr,         # [N, M, H_out, W_out]
    N, M, C, H, W, kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    H_out, W_out,
    BLOCK: tl.constexpr,
):
    """
    Each program computes a contiguous block of output elements.
    The block dimension is `BLOCK` (number of elements processed by a thread).
    """
    # Program id: position in the flattened output space
    pid = tl.program_id(0)

    # Compute start index in the flattened output tensor
    start_idx = pid * BLOCK
    end_idx   = start_idx + BLOCK

    # Offsets in flattened output
    offsets = tl.arange(0, BLOCK)

    # Mask to avoid out-of-bounds access
    mask = offsets + start_idx < N * M * H_out * W_out

    # Compute indices for each flattened output element
    out_idx = (offsets + start_idx) * mask  # zero out out-of-range indices

    # Convert flattened index to (n, m, oh, ow)
    oh = (out_idx // (M * W_out)) % H_out
    ow = (out_idx // M) % W_out
    m  = (out_idx // W_out) % M
    n  = out_idx // (M * H_out * W_out)

    # Compute the top-left corner of the receptive field
    h_start = oh * stride_h - pad_h
    w_start = ow * stride_w - pad_w

    # Accumulator for each thread
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Iterate over input channels and kernel spatial dimensions
    # Use tiling over (c, kh, kw)
    for c in range(0, C, BLOCK):
        c_tile = tl.arange(0, BLOCK)
        c_idx  = c + c_tile
        valid_c = c_idx < C

        for kh in range(0, kH, BLOCK):
            kh_tile = tl.arange(0, BLOCK)
            kh_idx  = kh + kh_tile
            valid_kh = kh_idx < kH

            for kw in range(0, kW, BLOCK):
                kw_tile = tl.arange(0, BLOCK)
                kw_idx  = kw + kw_tile
                valid_kw = kw_idx < kW

                # Compute input coordinates
                h_in = h_start + kh_idx * dilation_h
                w_in = w_start + kw_idx * dilation_w

                # Load input values
                inp_offsets = (
                    n * C * H * W
                    + c_idx * H * W
                    + h_in * W
                    + w_in
                )
                inp = tl.load(
                    input_ptr + inp_offsets,
                    mask=mask & valid_c & valid_kh & valid_kw & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W),
                    other=0.0,
                )

                # Load weight values
                wgt_offsets = (
                    m * C * kH * kW
                    + c_idx * kH * kW
                    + kh_idx * kW
                    + kw_idx
                )
                wgt = tl.load(
                    weight_ptr + wgt_offsets,
                    mask=mask & valid_c & valid_kh & valid_kw,
                    other=0.0,
                )

                # Accumulate
                acc += inp * wgt

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + m)
        acc += bias_val

    # Store the result
    out_offsets = (
        n * M * H_out * W_out
        + m * H_out * W_out
        + oh * W_out
        + ow
    )
    tl.store(output_ptr + out_offsets, acc, mask=mask)


# --------------------------------------------------
# Helper function to call the Triton kernel
# --------------------------------------------------
def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
):
    """
    Wrapper around the Triton convolution kernel.
    Assumes x and weight are on CUDA and contiguous.
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C, H, W = x.shape
    M, _, kH, kW = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # Compute output dimensions
    H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Allocate output tensor
    out = torch.empty((N, M, H_out, W_out), dtype=x.dtype, device=x.device)

    # Grid size: total number of output elements divided by BLOCK
    BLOCK = 128  # can be autotuned
    total_out = N * M * H_out * W_out
    grid = lambda meta: ((total_out + meta["BLOCK"] - 1) // meta["BLOCK"],)

    conv2d_kernel[grid](
        x, weight, bias, out,
        N, M, C, H, W, kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        H_out, W_out,
        BLOCK=BLOCK,
    )
    return out


# --------------------------------------------------
# Optimized model using Triton convolution
# --------------------------------------------------
class ModelNew(nn.Module):
    """
    Implements a 2D convolution using a custom Triton kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                device="cuda",
            ),
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )