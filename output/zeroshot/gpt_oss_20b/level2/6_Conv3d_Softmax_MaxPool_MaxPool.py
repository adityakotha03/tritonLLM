import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------------
# 3D convolution kernel
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def conv3d_kernel(
    inp_ptr,          # (N, C, D, H, W)
    filt_ptr,         # (C_out, C_in, kD, kH, kW)
    out_ptr,          # (N, C_out, D_out, H_out, W_out)
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program computes one output element (n, c_out, d_out, h_out, w_out).
    """
    idx = tl.program_id(0) * BLOCK_SIZE
    stride = BLOCK_SIZE

    # Compute total number of output elements
    total = N * C_out * D_out * H_out * W_out

    # Loop over assigned output elements
    for i in range(idx, total, stride):
        # Decode indices
        w_out = i % W_out
        h_out = (i // W_out) % H_out
        d_out = (i // (W_out * H_out)) % D_out
        c_out = (i // (W_out * H_out * D_out)) % C_out
        n = i // (W_out * H_out * D_out * C_out)

        # Load filter slice
        acc = tl.zeros([1], dtype=tl.float32)

        for c_in in range(C_in):
            for kd in range(kD):
                for kh in range(kH):
                    for kw in range(kW):
                        d_in = d_out * stride_d + kd
                        h_in = h_out * stride_h + kh
                        w_in = w_out * stride_w + kw
                        inp_idx = (
                            n * (C_in * D_in * H_in * W_in)
                            + c_in * (D_in * H_in * W_in)
                            + d_in * (H_in * W_in)
                            + h_in * W_in
                            + w_in
                        )
                        filt_idx = (
                            c_out * (C_in * kD * kH * kW)
                            + c_in * (kD * kH * kW)
                            + kd * (kH * kW)
                            + kh * kW
                            + kw
                        )
                        val = tl.load(inp_ptr + inp_idx, mask=True, other=0.0)
                        filt_val = tl.load(filt_ptr + filt_idx, mask=True, other=0.0)
                        acc += val * filt_val

        out_idx = (
            n * (C_out * D_out * H_out * W_out)
            + c_out * (D_out * H_out * W_out)
            + d_out * (H_out * W_out)
            + h_out * W_out
            + w_out
        )
        tl.store(out_ptr + out_idx, acc[0], mask=True)


def triton_conv3d(inp: torch.Tensor, filt: torch.Tensor, stride=1):
    """
    inp: (N, C_in, D_in, H_in, W_in)
    filt: (C_out, C_in, kD, kH, kW)
    """
    N, C_in, D_in, H_in, W_in = inp.shape
    C_out, _, kD, kH, kW = filt.shape
    stride_d = stride_h = stride_w = stride

    D_out = (D_in - kD) // stride_d + 1
    H_out = (H_in - kH) // stride_h + 1
    W_out = (W_in - kW) // stride_w + 1

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=inp.device, dtype=inp.dtype)

    grid = lambda meta: ( (N * C_out * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )

    conv3d_kernel[grid](
        inp, filt, out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        BLOCK_SIZE=meta["BLOCK_SIZE"],
    )
    return out


# ------------------------------------------------------------------
# Softmax kernel along channel dimension
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def softmax_channel_kernel(
    inp_ptr,          # (N, C, D, H, W)
    out_ptr,
    N, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes one spatial location (n, d, h, w) and the whole channel vector.
    """
    idx = tl.program_id(0) * BLOCK_SIZE
    stride = BLOCK_SIZE
    total = N * D * H * W

    for i in range(idx, total, stride):
        n = i // (D * H * W)
        d = (i // (H * W)) % D
        h = (i // W) % H
        w = i % W

        # Load channel vector
        # We process channels in chunks of BLOCK_SIZE
        max_val = tl.min(tl.full([1], float("-inf")))
        sum_exp = tl.zeros([1], dtype=tl.float32)

        # First pass to compute max
        for c in range(0, C, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = (c + offsets) < C
            idxs = (
                n * (C * D * H * W)
                + (c + offsets) * (D * H * W)
                + d * (H * W)
                + h * W
                + w
            )
            vals = tl.load(inp_ptr + idxs, mask=mask, other=0.0)
            max_val = tl.maximum(max_val, vals)

        # Second pass to compute sum of exp
        for c in range(0, C, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = (c + offsets) < C
            idxs = (
                n * (C * D * H * W)
                + (c + offsets) * (D * H * W)
                + d * (H * W)
                + h * W
                + w
            )
            vals = tl.load(inp_ptr + idxs, mask=mask, other=0.0)
            exp_vals = tl.exp(vals - max_val)
            sum_exp += tl.sum(exp_vals, axis=0, mask=mask)

        # Third pass to write normalized values
        for c in range(0, C, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = (c + offsets) < C
            idxs = (
                n * (C * D * H * W)
                + (c + offsets) * (D * H * W)
                + d * (H * W)
                + h * W
                + w
            )
            vals = tl.load(inp_ptr + idxs, mask=mask, other=0.0)
            exp_vals = tl.exp(vals - max_val)
            out_vals = exp_vals / sum_exp
            tl.store(out_ptr + idxs, out_vals, mask=mask)


def triton_softmax_channel(x: torch.Tensor):
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: ((N * D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softmax_channel_kernel[grid](
        x, out, N, C, D, H, W, BLOCK_SIZE=meta["BLOCK_SIZE"]
    )
    return out


# ------------------------------------------------------------------
# Max Pooling 3D kernel (kernel size 2, stride 2)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N", "C", "D_in", "H_in", "W_in"],
)
@triton.jit
def maxpool3d_kernel(
    inp_ptr,          # (N, C, D_in, H_in, W_in)
    out_ptr,
    N, C, D_in, H_in, W_in,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes one output element (n, c, d_out, h_out, w_out).
    """
    idx = tl.program_id(0) * BLOCK_SIZE
    stride = BLOCK_SIZE

    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2
    total = N * C * D_out * H_out * W_out

    for i in range(idx, total, stride):
        w_out = i % W_out
        h_out = (i // W_out) % H_out
        d_out = (i // (W_out * H_out)) % D_out
        c = (i // (W_out * H_out * D_out)) % C
        n = i // (W_out * H_out * D_out * C)

        d_in = d_out * 2
        h_in = h_out * 2
        w_in = w_out * 2

        # 8-element window
        vals = tl.zeros([8], dtype=tl.float32)
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    idxs = (
                        n * (C * D_in * H_in * W_in)
                        + c * (D_in * H_in * W_in)
                        + (d_in + dz) * (H_in * W_in)
                        + (h_in + dy) * W_in
                        + (w_in + dx)
                    )
                    vals[dz * 4 + dy * 2 + dx] = tl.load(inp_ptr + idxs, mask=True, other=float("-inf"))

        max_val = tl.reduce_max(vals, axis=0)
        out_idx = (
            n * (C * D_out * H_out * W_out)
            + c * (D_out * H_out * W_out)
            + d_out * (H_out * W_out)
            + h_out * W_out
            + w_out
        )
        tl.store(out_ptr + out_idx, max_val, mask=True)


def triton_maxpool3d(x: torch.Tensor, kernel_size=2, stride=2):
    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in // stride
    H_out = H_in // stride
    W_out = W_in // stride
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: ((N * C * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    maxpool3d_kernel[grid](
        x, out, N, C, D_in, H_in, W_in, BLOCK_SIZE=meta["BLOCK_SIZE"]
    )
    return out


# ------------------------------------------------------------------
# New model using custom Triton kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Create filter tensor as a learnable parameter
        k = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k, k, k, device="cuda")
        )
        # Bias is optional; omitted for brevity

    def forward(self, x):
        # Convolution
        conv_out = triton_conv3d(x, self.weight, stride=1)

        # Softmax over channel dimension (dim=1)
        softmax_out = triton_softmax_channel(conv_out)

        # First max pool
        pool1_out = triton_maxpool3d(softmax_out, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)

        # Second max pool
        pool2_out = triton_maxpool3d(pool1_out, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)

        return pool2_out