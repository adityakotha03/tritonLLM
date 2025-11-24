import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------
# 1‑x‑1 convolution (matrix multiplication)
# ------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=16),
    ],
    key=["M", "N"],
)
@triton.jit
def conv1x1_kernel(
    X_ptr,   # (M, K) input
    W_ptr,   # (N, K) weight
    out_ptr, # (M, N) output
    M, N, K,
    stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    block_m = pid_x * BLOCK_M
    block_n = pid_y * BLOCK_N

    offs_m = block_m + tl.arange(0, BLOCK_M)
    offs_n = block_n + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_N):
        X = tl.load(X_ptr + (offs_m[:, None] * stride) + (k + tl.arange(0, BLOCK_N)[None, :]),
                    mask=offs_m[:, None] < M, other=0.0)
        W = tl.load(W_ptr + (offs_n[None, :] * stride) + (k + tl.arange(0, BLOCK_N)[None, :]),
                    mask=offs_n[None, :] < N, other=0.0)
        acc += X.to(tl.float32) @ W.to(tl.float32).T

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + (offs_m[:, None] * stride) + offs_n[None, :], acc.to(tl.float32), mask=mask)


def conv1x1_torch(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    X: (batch, in_ch, H, W)
    W: (out_ch, in_ch, 1, 1)
    """
    B, C_in, H, W_ = X.shape
    C_out = W.shape[0]
    X_reshaped = X.view(B, C_in, -1).transpose(1, 2)          # (B, H*W, C_in)
    W_reshaped = W.view(C_out, C_in)                          # (C_out, C_in)

    # Transpose to (B, C_in, H*W) for GEMM
    X_mat = X_reshaped.reshape(B * H * W_, C_in)              # (B*HW, C_in)
    W_mat = W_reshaped                                     # (C_out, C_in)

    # Launch Triton kernel
    out_mat = torch.empty(B * H * W_, C_out, device=X.device, dtype=torch.float32)

    grid = lambda meta: (
        (out_mat.shape[0] + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (out_mat.shape[1] + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )
    conv1x1_kernel[grid](X_mat, W_mat, out_mat, X_mat.shape[0], out_mat.shape[1], C_in,
                         stride=1, BLOCK_M=meta["BLOCK_M"], BLOCK_N=meta["BLOCK_N"])

    out = out_mat.view(B, H, W_, C_out).permute(0, 3, 1, 2)  # (B, C_out, H, W)
    return out


# ------------------------------------
# 3‑x‑3 and 5‑x‑5 convolution (im2col + GEMM)
# ------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128, "BLOCK_K": 128}, num_warps=16),
    ],
    key=["H", "W", "K"],
)
@triton.jit
def conv2d_kernel(
    X_ptr,  # (B, C_in, H_in, W_in)
    W_ptr,  # (C_out, C_in, KH, KW)
    out_ptr,  # (B, C_out, H_out, W_out)
    B, C_in, H_in, W_in, C_out, KH, KW,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bid = tl.program_id(0)
    cid = tl.program_id(1)
    hid = tl.program_id(2)
    wid = tl.program_id(3)

    block_h = hid * BLOCK_H
    block_w = wid * BLOCK_W

    offs_h = block_h + tl.arange(0, BLOCK_H)
    offs_w = block_w + tl.arange(0, BLOCK_W)

    # Compute output positions
    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    mask_h = offs_h < H_out
    mask_w = offs_w < W_out

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    for k in range(C_in):
        # Load input tile
        X_tile = tl.load(
            X_ptr + (bid * C_in * H_in * W_in)
            + (k * H_in * W_in)
            + ((offs_h * stride_h - pad_h)[:, None] + tl.arange(0, KH))
            + ((offs_w * stride_w - pad_w)[None, :] + tl.arange(0, KW)),
            mask=(offs_h[:, None] * stride_h - pad_h + tl.arange(0, KH) < H_in) &
                 (offs_w[None, :] * stride_w - pad_w + tl.arange(0, KW) < W_in),
            other=0.0,
        )
        # Load kernel tile
        W_tile = tl.load(
            W_ptr + (cid * C_in * KH * KW)
            + (k * KH * KW)
            + (tl.arange(0, KH)[:, None] + tl.arange(0, KW)[None, :]),
            mask=tl.arange(0, KH)[:, None] < KH,
            other=0.0,
        )
        acc += X_tile.to(tl.float32) @ W_tile.to(tl.float32).T

    mask = mask_h[:, None] & mask_w[None, :]
    tl.store(
        out_ptr + (bid * C_out * H_out * W_out)
        + (cid * H_out * W_out)
        + (offs_h[:, None] * H_out + offs_w[None, :]),
        acc.to(tl.float32),
        mask=mask,
    )


def conv2d_torch(X: torch.Tensor, W: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor:
    B, C_in, H_in, W_in = X.shape
    C_out, _, KH, KW = W.shape
    stride_h = stride_w = 1

    H_out = (H_in + 2 * padding - KH) // stride_h + 1
    W_out = (W_in + 2 * padding - KW) // stride_w + 1

    out = torch.empty(B, C_out, H_out, W_out, device=X.device, dtype=torch.float32)

    grid = lambda meta: (
        B,
        C_out,
        (H_out + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W_out + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )
    conv2d_kernel[grid](
        X, W, out, B, C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w, padding, padding,
        BLOCK_H=meta["BLOCK_H"], BLOCK_W=meta["BLOCK_W"], BLOCK_K=meta["BLOCK_K"],
    )
    return out


# ------------------------------------
# Max‑pooling + 1‑x‑1 projection
# ------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=8),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=16),
    ],
    key=["H", "W"],
)
@triton.jit
def maxpool1x1_kernel(
    X_ptr,   # (B, C_in, H, W)
    W_ptr,   # (C_out, C_in, 1, 1)
    out_ptr, # (B, C_out, H, W)
    B, C_in, H, W, C_out,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    bid = tl.program_id(0)
    cid = tl.program_id(1)
    hid = tl.program_id(2)
    wid = tl.program_id(3)

    block_h = hid * BLOCK_H
    block_w = wid * BLOCK_W

    offs_h = block_h + tl.arange(0, BLOCK_H)
    offs_w = block_w + tl.arange(0, BLOCK_W)

    mask_h = offs_h < H
    mask_w = offs_w < W

    # Max‑pool over 3×3 window with stride 1, padding 1
    pooled = tl.full([BLOCK_H, BLOCK_W], -1e9, dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h = offs_h + kh
            w = offs_w + kw
            mask = (h >= 0) & (h < H) & (w >= 0) & (w < W)
            val = tl.load(
                X_ptr + (bid * C_in * H * W) + (C_in * h[:, None] * W + C_in * w[None, :]),
                mask=mask[:, None] & mask[None, :],
                other=-1e9,
            )
            pooled = tl.max(pooled, val)

    # 1‑x‑1 projection
    proj = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for c_in in range(C_in):
        X = pooled.to(tl.float32)
        W_t = tl.load(
            W_ptr + (cid * C_in) + c_in,
            mask=True,
            other=0.0,
        )
        proj += X * W_t

    mask = mask_h[:, None] & mask_w[None, :]
    tl.store(
        out_ptr + (bid * C_out * H * W) + (cid * H * W) + (offs_h[:, None] * W + offs_w[None, :]),
        proj.to(tl.float32),
        mask=mask,
    )


def maxpool1x1_torch(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    B, C_in, H, W_ = X.shape
    C_out = W.shape[0]
    out = torch.empty(B, C_out, H, W_, device=X.device, dtype=torch.float32)

    grid = lambda meta: (
        B,
        C_out,
        (H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W_ + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )
    maxpool1x1_kernel[grid](
        X, W, out, B, C_in, H, W_, C_out,
        BLOCK_H=meta["BLOCK_H"], BLOCK_W=meta["BLOCK_W"],
    )
    return out


# ------------------------------------
# Model using the Triton kernels
# ------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()

        # 1×1 conv
        self.conv1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3×3 conv branch
        self.branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        # 5×5 conv branch
        self.branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        # Pool branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

        # Store weights for Triton kernels
        self._conv1x1_weight = self.conv1x1.weight
        self._branch3x3_reduce_weight = self.branch3x3_reduce.weight
        self._branch3x3_weight = self.branch3x3.weight
        self._branch5x5_reduce_weight = self.branch5x5_reduce.weight
        self._branch5x5_weight = self.branch5x5.weight
        self._branch_pool_weight = self.branch_pool[1].weight

    def forward(self, x):
        # 1×1 conv branch
        branch1x1 = conv1x1_torch(x, self._conv1x1_weight)

        # 3×3 conv branch
        reduced = conv1x1_torch(x, self._branch3x3_reduce_weight)
        branch3x3 = conv2d_torch(reduced, self._branch3x3_weight, kernel_size=3, padding=1)

        # 5×5 conv branch
        reduced5 = conv1x1_torch(x, self._branch5x5_reduce_weight)
        branch5x5 = conv2d_torch(reduced5, self._branch5x5_weight, kernel_size=5, padding=2)

        # Pool branch
        pool = self.branch_pool[0](x)  # MaxPool2d (standard PyTorch)
        branch_pool = maxpool1x1_torch(pool, self._branch_pool_weight)

        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)