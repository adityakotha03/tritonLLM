import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernel for a depth‑wise 2D convolution (stride=1, dilation=1, no bias)
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['in_H', 'in_W', 'kH', 'kW'],
)
@triton.jit
def depthwise_conv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch,
    channels,
    in_H,
    in_W,
    out_H,
    out_W,
    kH,
    kW,
    stride_H,
    stride_W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Performs a depth‑wise convolution with a single input channel per output channel.
    Each program processes a contiguous block of output elements across the
    batch, channel and spatial dimensions.
    """
    pid = tl.program_id(0)
    # Total number of output elements
    total_out = batch * channels * out_H * out_W
    offset = pid * BLOCK_SIZE
    # Indices of elements processed by this program
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < total_out

    # Decode linear indices into 4‑D coordinates
    # batch, channel, out_H, out_W
    batch_id = indices // (channels * out_H * out_W)
    rem = indices % (channels * out_H * out_W)
    channel_id = rem // (out_H * out_W)
    rem2 = rem % (out_H * out_W)
    out_h_id = rem2 // out_W
    out_w_id = rem2 % out_W

    # Compute the top‑left corner of the kernel window in the input
    in_h_start = out_h_id * stride_H
    in_w_start = out_w_id * stride_W

    # Load the kernel weights for this channel
    # We assume weight shape: (channels, 1, kH, kW)
    w_offset = channel_id * kH * kW
    w = tl.load(w_ptr + w_offset + tl.arange(0, kH * kW), mask=mask, other=0.0)
    w = tl.reshape(w, (kH, kW))

    # Accumulate convolution result
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for kh in range(kH):
        for kw in range(kW):
            # Compute input indices
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            # Load input value
            # Input shape: (batch, channels, in_H, in_W)
            inp_offset = (
                batch_id * channels * in_H * in_W
                + channel_id * in_H * in_W
                + in_h * in_W
                + in_w
            )
            inp = tl.load(x_ptr + inp_offset, mask=mask, other=0.0)
            acc += inp * w[kh, kw]
    # Store output
    out_offset = (
        batch_id * channels * out_H * out_W
        + channel_id * out_H * out_W
        + out_h_id * out_W
        + out_w_id
    )
    tl.store(out_ptr + out_offset, acc, mask=mask)

# -------------------------------------------------------------
# Triton wrapper that replaces nn.Conv2d (depth‑wise)
# -------------------------------------------------------------
def triton_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, stride: tuple, padding: tuple):
    """
    x:      (N, C, H, W)
    weight: (C, 1, kH, kW)
    stride: (sH, sW)
    padding: (pH, pW)   (only zero padding is supported)
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    N, C, H, W = x.shape
    kH, kW = weight.shape[2], weight.shape[3]
    sH, sW = stride
    pH, pW = padding

    # Output dimensions
    out_H = (H + 2 * pH - kH) // sH + 1
    out_W = (W + 2 * pW - kW) // sW + 1
    out = torch.empty((N, C, out_H, out_W), dtype=x.dtype, device=x.device)

    # Pad input if needed
    if pH > 0 or pW > 0:
        x_pad = torch.nn.functional.pad(x, (pW, pW, pH, pH))
    else:
        x_pad = x

    # Launch kernel
    grid = lambda meta: ((N * C * out_H * out_W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    depthwise_conv_kernel[grid](
        x_pad, weight, out,
        N, C,
        H + 2 * pH, W + 2 * pW,
        out_H, out_W,
        kH, kW,
        sH, sW,
        BLOCK_SIZE=128,
    )
    return out

# -------------------------------------------------------------
# Optimized model that uses the Triton depth‑wise conv kernel
# -------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(ModelNew, self).__init__()
        assert dilation_h == 1 and dilation_w == 1, "Dilation >1 not supported in Triton impl"
        assert bias == False, "Bias not supported in Triton impl"

        # Store parameters for the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)

        # Create weight parameter (depth‑wise conv)
        self.weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size_h, kernel_size_w)
        )
        # Note: groups = in_channels ensures depth‑wise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(
            x,
            self.weight,
            self.stride,
            self.padding,
        )