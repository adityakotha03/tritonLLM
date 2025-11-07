import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3, x_stride_4,
    w_stride_0, w_stride_1, w_stride_2, w_stride_3, w_stride_4,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3, out_stride_4,
    batch_size, in_channels, out_channels, D, H, W, kD, kH, kW,
    stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Define block indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Block size
    bs_d = BLOCK_SIZE_D
    bs_h = BLOCK_SIZE_H
    bs_w = BLOCK_SIZE_W
    bs_c = BLOCK_SIZE_C

    # Compute offsets
    d_start = pid_d * bs_d
    h_start = pid_h * bs_h
    w_start = pid_w * bs_w
    c_start = pid_c * bs_c

    # Define indices
    d_offsets = tl.arange(0, bs_d)
    h_offsets = tl.arange(0, bs_h)
    w_offsets = tl.arange(0, bs_w)
    c_offsets = tl.arange(0, bs_c)

    # Compute input and output indices
    # Output indices
    out_d = d_start + d_offsets
    out_h = h_start + h_offsets
    out_w = w_start + w_offsets
    out_c = c_start + c_offsets

    # Input indices (after padding)
    in_d = out_d * stride_d - pad_d
    in_h = out_h * stride_h - pad_h
    in_w = out_w * stride_w - pad_w

    # Kernel indices
    k_d = tl.arange(0, kD)
    k_h = tl.arange(0, kH)
    k_w = tl.arange(0, kW)
    k_c = tl.arange(0, in_channels)

    # Load weights (channel dimension is fixed)
    w_ptrs = w_ptr + (
        out_c[:, None, None, None] * w_stride_1 +
        k_c[None, :, None, None] * w_stride_0 +
        k_d[None, None, :, None] * w_stride_2 +
        k_h[None, None, None, :] * w_stride_3 +
        k_w[None, None, None, :] * w_stride_4
    )

    w_vals = tl.load(w_ptrs, mask=(out_c[:, None, None, None] < out_channels) & (k_c[None, :, None, None] < in_channels) & (k_d[None, None, :, None] < kD) & (k_h[None, None, None, :] < kH) & (k_w[None, None, None, :] < kW), other=0.0)

    # Initialize accumulator
    acc = tl.zeros((bs_c, bs_d, bs_h, bs_w), dtype=tl.float32)

    # Loop over input channels and kernel
    for c_idx in range(0, in_channels, bs_c):
        # Compute input channel offset
        c_offset = c_idx
        c_mask = c_offset + c_offsets < in_channels

        # Compute input indices
        in_c = c_offset + c_offsets
        in_c_mask = c_offset + c_offsets < in_channels

        # Load input
        x_ptrs = x_ptr + (
            pid_batch * x_stride_0 +
            in_c[:, None, None, None] * x_stride_1 +
            in_d[None, :, None, None] * x_stride_2 +
            in_h[None, None, :, None] * x_stride_3 +
            in_w[None, None, None, :] * x_stride_4
        )

        # Load input with masking
        x_vals = tl.load(x_ptrs, mask=(in_c[:, None, None, None] < in_channels) & (in_d[None, :, None, None] < D) & (in_h[None, None, :, None] < H) & (in_w[None, None, None, :] < W) & (in_d[None, :, None, None] >= 0) & (in_h[None, None, :, None] >= 0) & (in_w[None, None, None, :] >= 0), other=0.0)

        # Perform convolution (reduce over kernel and input channel dims)
        acc += tl.dot(w_vals, x_vals, allow_tf32=False)

    # Apply output mask
    out_mask = (out_d < D) & (out_h < H) & (out_w < W) & (out_c < out_channels)

    # Store result
    out_ptrs = out_ptr + (
        pid_batch * out_stride_0 +
        out_c[:, None, None, None] * out_stride_1 +
        out_d[None, :, None, None] * out_stride_2 +
        out_h[None, None, :, None] * out_stride_3 +
        out_w[None, None, None, :] * out_stride_4
    )

    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def mish_tanh_kernel(
    x_ptr, out_ptr,
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply Mish
    # f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    exp_x = tl.exp(x)
    softplus = tl.log(1.0 + exp_x)
    tanh_softplus = tl.tanh(softplus)
    mish = x * tanh_softplus

    # Apply Tanh
    out = tl.tanh(mish)

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv3d_mish_tanh(x, weight, stride_d=1, stride_h=1, stride_w=1, pad_d=0, pad_h=0, pad_w=0):
    """
    Optimized 3D convolution with Mish and Tanh activations using Triton kernels.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kD, kH, kW = weight.shape

    # Output dimensions
    D_out = (D + 2 * pad_d - kD) // stride_d + 1
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

    # Get strides
    x_stride_0, x_stride_1, x_stride_2, x_stride_3, x_stride_4 = x.stride()
    w_stride_0, w_stride_1, w_stride_2, w_stride_3, w_stride_4 = weight.stride()
    out_stride_0, out_stride_1, out_stride_2, out_stride_3, out_stride_4 = out.stride()

    # Configure kernel launch
    BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W = 32, 32, 32
    BLOCK_SIZE_C = 32

    # Grid: (batch, out_channels, D_out, H_out, W_out)
    grid = lambda meta: (
        meta["batch_size"],
        (meta["out_channels"] + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        (meta["D_out"] + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (meta["H_out"] + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (meta["W_out"] + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"]
    )

    # Launch kernel
    conv3d_kernel[grid](
        x, weight, out,
        x_stride_0, x_stride_1, x_stride_2, x_stride_3, x_stride_4,
        w_stride_0, w_stride_1, w_stride_2, w_stride_3, w_stride_4,
        out_stride_0, out_stride_1, out_stride_2, out_stride_3, out_stride_4,
        batch_size, in_channels, out_channels, D, H, W, kD, kH, kW,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    # Apply Mish + Tanh fusion
    n_elements = out.numel()
    BLOCK_SIZE = 1024  # Optimized block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mish_tanh_kernel[grid](out, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

    def forward(self, x):
        return triton_conv3d_mish_tanh(
            x,
            self.weight,
            stride_d=self.stride,
            stride_h=self.stride,
            stride_w=self.stride,
            pad_d=self.padding,
            pad_h=self.padding,
            pad_w=self.padding
        )