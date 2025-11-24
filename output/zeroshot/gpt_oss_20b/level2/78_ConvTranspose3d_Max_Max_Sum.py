import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel for 3D transposed convolution
# ------------------------------------------------------------------
@triton.jit
def conv_transpose3d_kernel(
    in_ptr,
    weight_ptr,
    out_ptr,
    in_D, in_H, in_W,
    out_D, out_H, out_W,
    in_C, out_C,
    kD, kH, kW,
    stride_D, stride_H, stride_W,
    pad_D, pad_H, pad_W,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: one program per output voxel and output channel
    batch_id   = tl.program_id(0)
    out_c_id   = tl.program_id(1)
    out_d_start = tl.program_id(2) * BLOCK_D
    out_h_start = tl.program_id(3) * BLOCK_H
    out_w_start = tl.program_id(4) * BLOCK_W

    # Indices of the output block
    d_offsets = tl.arange(0, BLOCK_D) + out_d_start
    h_offsets = tl.arange(0, BLOCK_H) + out_h_start
    w_offsets = tl.arange(0, BLOCK_W) + out_w_start

    mask_d = d_offsets < out_D
    mask_h = h_offsets < out_H
    mask_w = w_offsets < out_W

    # Iterate over kernel window
    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                # Corresponding input positions
                in_d = (d_offsets - pad_D + kd) // stride_D
                in_h = (h_offsets - pad_H + kh) // stride_H
                in_w = (w_offsets - pad_W + kw) // stride_W

                mask_in_d = (in_d >= 0) & (in_d < in_D)
                mask_in_h = (in_h >= 0) & (in_h < in_H)
                mask_in_w = (in_w >= 0) & (in_w < in_W)

                full_mask = mask_d & mask_h & mask_w & mask_in_d & mask_in_h & mask_in_w
                if tl.all(~full_mask):
                    continue

                # Load input slice
                in_idx = (
                    batch_id * in_C * in_D * in_H * in_W
                    + tl.arange(0, in_C)[:, None, None] * in_D * in_H * in_W
                    + in_d[None, :, None] * in_H * in_W
                    + in_h[None, None, :] * in_W
                    + in_w[None, None, None]
                )
                in_vals = tl.load(in_ptr + in_idx, mask=full_mask[None, :, :], other=0.0)

                # Load weight
                weight_idx = (
                    out_c_id * in_C * kD * kH * kW
                    + tl.arange(0, in_C)[:, None, None] * kD * kH * kW
                    + kd * kH * kW
                    + kh * kW
                    + kw
                )
                weight_vals = tl.load(weight_ptr + weight_idx, mask=None)

                acc += in_vals * weight_vals[None, :, None, None]

    # Store output
    out_idx = (
        batch_id * out_C * out_D * out_H * out_W
        + out_c_id * out_D * out_H * out_W
        + d_offsets[:, None, None] * out_H * out_W
        + h_offsets[None, :, None] * out_W
        + w_offsets[None, None, :]
    )
    tl.store(out_ptr + out_idx, acc, mask=full_mask)

# ------------------------------------------------------------------
# Triton kernel for 3D max pooling
# ------------------------------------------------------------------
@triton.jit
def maxpool3d_kernel(
    in_ptr,
    out_ptr,
    in_D, in_H, in_W,
    out_D, out_H, out_W,
    kD, kH, kW,
    stride_D, stride_H, stride_W,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    batch_id   = tl.program_id(0)
    channel_id = tl.program_id(1)
    out_d_start = tl.program_id(2) * BLOCK_D
    out_h_start = tl.program_id(3) * BLOCK_H
    out_w_start = tl.program_id(4) * BLOCK_W

    d_offsets = tl.arange(0, BLOCK_D) + out_d_start
    h_offsets = tl.arange(0, BLOCK_H) + out_h_start
    w_offsets = tl.arange(0, BLOCK_W) + out_w_start

    mask_d = d_offsets < out_D
    mask_h = h_offsets < out_H
    mask_w = w_offsets < out_W

    acc = tl.full([BLOCK_D, BLOCK_H, BLOCK_W], -float("inf"), dtype=tl.float32)

    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                in_d = d_offsets * stride_D + kd
                in_h = h_offsets * stride_H + kh
                in_w = w_offsets * stride_W + kw

                mask_in_d = (in_d >= 0) & (in_d < in_D)
                mask_in_h = (in_h >= 0) & (in_h < in_H)
                mask_in_w = (in_w >= 0) & (in_w < in_W)

                full_mask = mask_d & mask_h & mask_w & mask_in_d & mask_in_h & mask_in_w
                if tl.all(~full_mask):
                    continue

                in_idx = (
                    batch_id * in_D * in_H * in_W
                    + channel_id * in_D * in_H * in_W
                    + in_d[None, :, None] * in_H * in_W
                    + in_h[None, None, :] * in_W
                    + in_w[None, None, None]
                )
                vals = tl.load(in_ptr + in_idx, mask=full_mask[None, :, :], other=-float("inf"))
                acc = tl.maximum(acc, vals[None, :, None, None])

    out_idx = (
        batch_id * out_D * out_H * out_W
        + channel_id * out_D * out_H * out_W
        + d_offsets[:, None, None] * out_H * out_W
        + h_offsets[None, :, None] * out_W
        + w_offsets[None, None, :]
    )
    tl.store(out_ptr + out_idx, acc, mask=full_mask)

# ------------------------------------------------------------------
# Triton kernel for channel-wise sum
# ------------------------------------------------------------------
@triton.jit
def sum_channels_kernel(
    in_ptr,
    out_ptr,
    D, H, W,
    C,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    batch_id   = tl.program_id(0)
    d_start = tl.program_id(1) * BLOCK_D
    h_start = tl.program_id(2) * BLOCK_H
    w_start = tl.program_id(3) * BLOCK_W

    d_offsets = tl.arange(0, BLOCK_D) + d_start
    h_offsets = tl.arange(0, BLOCK_H) + h_start
    w_offsets = tl.arange(0, BLOCK_W) + w_start

    mask_d = d_offsets < D
    mask_h = h_offsets < H
    mask_w = w_offsets < W

    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=tl.float32)

    for c in range(C):
        idx = (
            batch_id * C * D * H * W
            + c * D * H * W
            + d_offsets[:, None, None] * H * W
            + h_offsets[None, :, None] * W
            + w_offsets[None, None, :]
        )
        vals = tl.load(in_ptr + idx, mask=mask_d & mask_h & mask_w, other=0.0)
        acc += vals[None, :, None, None]

    out_idx = (
        batch_id * D * H * W
        + d_offsets[:, None, None] * H * W
        + h_offsets[None, :, None] * W
        + w_offsets[None, None, :]
    )
    tl.store(out_ptr + out_idx, acc, mask=mask_d & mask_h & mask_w)

# ------------------------------------------------------------------
# Wrapper functions that launch the Triton kernels
# ------------------------------------------------------------------
def triton_conv_transpose3d(x, weight, stride, padding, kernel_size):
    batch, in_c, in_d, in_h, in_w = x.shape
    kD, kH, kW = kernel_size
    stride_D, stride_H, stride_W = stride
    pad_D, pad_H, pad_W = padding

    # Output shape
    out_d = (in_d - 1) * stride_D - 2 * pad_D + kD
    out_h = (in_h - 1) * stride_H - 2 * pad_H + kH
    out_w = (in_w - 1) * stride_W - 2 * pad_W + kW
    out_c = weight.shape[0]

    out = torch.empty((batch, out_c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)

    BLOCK_D, BLOCK_H, BLOCK_W = 16, 16, 16
    grid = lambda meta: (
        batch,
        out_c,
        (out_d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (out_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    conv_transpose3d_kernel[grid](
        x,
        weight,
        out,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        in_c, out_c,
        kD, kH, kW,
        stride_D, stride_H, stride_W,
        pad_D, pad_H, pad_W,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out

def triton_maxpool3d(x, kernel_size, stride):
    batch, c, d, h, w = x.shape
    kD, kH, kW = kernel_size
    stride_D, stride_H, stride_W = stride

    out_d = (d - kD) // stride_D + 1
    out_h = (h - kH) // stride_H + 1
    out_w = (w - kW) // stride_W + 1

    out = torch.empty((batch, c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)

    BLOCK_D, BLOCK_H, BLOCK_W = 16, 16, 16
    grid = lambda meta: (
        batch,
        c,
        (out_d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (out_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    maxpool3d_kernel[grid](
        x,
        out,
        d, h, w,
        out_d, out_h, out_w,
        kD, kH, kW,
        stride_D, stride_H, stride_W,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out

def triton_sum_channels(x):
    batch, c, d, h, w = x.shape
    out = torch.empty((batch, 1, d, h, w), device=x.device, dtype=x.dtype)

    BLOCK_D, BLOCK_H, BLOCK_W = 32, 32, 32
    grid = lambda meta: (
        batch,
        (d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    sum_channels_kernel[grid](
        x,
        out,
        d, h, w,
        c,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out

# ------------------------------------------------------------------
# Optimized model using Triton kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that replaces ConvTranspose3d, MaxPool3d and channel sum
    with custom Triton kernels for higher throughput on A100.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        # Store parameters for Triton kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)

        # Trainable weight for transposed convolution
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                kernel_size,
                device="cuda",
                dtype=torch.float32,
            )
        )

    def forward(self, x):
        # Transposed convolution
        x = triton_conv_transpose3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            kernel_size=self.kernel_size,
        )
        # First max pool (kernel size 2, stride 2)
        x = triton_maxpool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Second max pool (kernel size 3, stride 3)
        x = triton_maxpool3d(x, kernel_size=(3, 3, 3), stride=(3, 3, 3))
        # Sum over channels
        x = triton_sum_channels(x)
        return x