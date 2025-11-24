import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------
# 1.  Triton kernel: 2‑D convolution (kernel=5, stride=1) + ReLU
# ------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_X": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_X": 256}, num_warps=8),
    ],
    key=["N", "C", "H", "W", "K", "KH", "KW"],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr,        # [N, C, H, W] input
    W_ptr,        # [K, C, KH, KW] kernel
    Y_ptr,        # [N, K, H_out, W_out] output
    N, C, H, W,   # input shape
    K, KH, KW,    # kernel shape
    stride_h, stride_w,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr = 32,
):
    # Compute output spatial dimensions
    H_out = (H - KH) // stride_h + 1
    W_out = (W - KW) // stride_w + 1

    # grid: (N, K, H_out, W_out)
    batch_idx = tl.program_id(0)
    k_idx    = tl.program_id(1)
    h_idx    = tl.program_id(2)
    w_idx    = tl.program_id(3)

    # Each thread processes a tile of output pixels
    for h_tile in range(0, H_out, BLOCK_SIZE_X):
        for w_tile in range(0, W_out, BLOCK_SIZE_X):
            h_off = h_tile + tl.arange(0, BLOCK_SIZE_X)
            w_off = w_tile + tl.arange(0, BLOCK_SIZE_X)
            mask_h = h_off < H_out
            mask_w = w_off < W_out
            mask = mask_h[:, None] & mask_w[None, :]

            acc = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_X], dtype=tl.float32)

            for c in range(C):
                for kh in range(KH):
                    for kw in range(KW):
                        h_in = h_off * stride_h + kh
                        w_in = w_off * stride_w + kw
                        inp_ptr = X_ptr + (batch_idx * C * H * W
                                           + c * H * W
                                           + h_in * W
                                           + w_in)
                        ker_ptr = W_ptr + (k_idx * C * KH * KW
                                           + c * KH * KW
                                           + kh * KW
                                           + kw)
                        inp = tl.load(inp_ptr, mask=mask, other=0.0)
                        ker = tl.load(ker_ptr)
                        acc += inp * ker

            # ReLU
            acc = tl.maximum(acc, 0.0)

            out_ptr = Y_ptr + (batch_idx * K * H_out * W_out
                               + k_idx * H_out * W_out
                               + h_tile * W_out
                               + w_tile)
            tl.store(out_ptr, acc, mask=mask)


def conv2d_relu(x: torch.Tensor, w: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Triton based conv2d + ReLU (kernel size 5, stride 1)."""
    N, C, H, W = x.shape
    K, _, KH, KW = w.shape
    assert KH == 5 and KW == 5
    assert stride == 1

    H_out = H - KH + 1
    W_out = W - KW + 1

    y = torch.empty((N, K, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        (N, K, (H_out + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"],
         (W_out + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"]),
    )

    conv2d_relu_kernel[grid](
        x,
        w,
        y,
        N, C, H, W,
        K, KH, KW,
        stride, stride,
        BLOCK_SIZE_X=meta["BLOCK_SIZE_X"],
    )
    return y


# ------------------------------------------------------
# 2.  Triton kernel: Max‑pool 2‑D (kernel=2, stride=2)
# ------------------------------------------------------
@triton.jit
def maxpool2d_kernel(
    X_ptr,       # [N, C, H, W]
    Y_ptr,       # [N, C, H_out, W_out]
    N, C, H, W,
    stride_h, stride_w,
    BLOCK_SIZE_X: tl.constexpr,
):
    H_out = H // stride_h
    W_out = W // stride_w

    batch_idx = tl.program_id(0)
    c_idx     = tl.program_id(1)
    h_idx     = tl.program_id(2)
    w_idx     = tl.program_id(3)

    for h_tile in range(0, H_out, BLOCK_SIZE_X):
        for w_tile in range(0, W_out, BLOCK_SIZE_X):
            h_off = h_tile + tl.arange(0, BLOCK_SIZE_X)
            w_off = w_tile + tl.arange(0, BLOCK_SIZE_X)
            mask_h = h_off < H_out
            mask_w = w_off < W_out
            mask = mask_h[:, None] & mask_w[None, :]

            acc = tl.full([BLOCK_SIZE_X, BLOCK_SIZE_X], float('-inf'), dtype=tl.float32)

            for kh in range(stride_h):
                for kw in range(stride_w):
                    h_in = h_off * stride_h + kh
                    w_in = w_off * stride_w + kw
                    inp_ptr = X_ptr + (batch_idx * C * H * W
                                       + c_idx * H * W
                                       + h_in * W
                                       + w_in)
                    inp = tl.load(inp_ptr, mask=mask, other=float('-inf'))
                    acc = tl.maximum(acc, inp)

            out_ptr = Y_ptr + (batch_idx * C * H_out * W_out
                               + c_idx * H_out * W_out
                               + h_tile * W_out
                               + w_tile)
            tl.store(out_ptr, acc, mask=mask)


def maxpool2d(x: torch.Tensor, stride: int = 2) -> torch.Tensor:
    N, C, H, W = x.shape
    H_out = H // stride
    W_out = W // stride

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        (N, C,
         (H_out + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"],
         (W_out + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"]),
    )

    maxpool2d_kernel[grid](
        x,
        y,
        N, C, H, W,
        stride, stride,
        BLOCK_SIZE_X=meta["BLOCK_SIZE_X"],
    )
    return y


# ------------------------------------------------------
# 3.  Triton kernel: Linear + ReLU (matrix multiplication)
# ------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 32}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_relu_kernel(
    A_ptr,   # [M, K]
    B_ptr,   # [K, N]
    C_ptr,   # [M, N]
    bias_ptr,  # [N]
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = tl.program_id(1) * N + tl.arange(0, N)

    mask_row = row < M
    mask_col = col < N

    acc = tl.zeros([BLOCK_SIZE_M, N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr + (row[:, None] * K + k + tl.arange(0, BLOCK_SIZE_K)), mask=mask_row[:, None], other=0.0)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + col[None, :], mask=mask_col[None, :], other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + col, mask=mask_col, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    tl.store(C_ptr + (row[:, None] * N + col[None, :]), acc, mask=mask_row[:, None] & mask_col[None, :])


def linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        ( (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
          (N + 1) // 1 ),  # one thread per column
    )

    linear_relu_kernel[grid](
        x,
        weight.t(),
        y,
        bias,
        M, N, K,
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return y


# ------------------------------------------------------
# 4.  Model with Triton kernels
# ------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes: int):
        super(ModelNew, self).__init__()
        # Parameters
        self.conv1_weight = nn.Parameter(torch.randn(6, 1, 5, 5, device="cuda"))
        self.conv2_weight = nn.Parameter(torch.randn(16, 6, 5, 5, device="cuda"))

        self.fc1_weight = nn.Parameter(torch.randn(120, 16 * 5 * 5, device="cuda"))
        self.fc1_bias   = nn.Parameter(torch.randn(120, device="cuda"))

        self.fc2_weight = nn.Parameter(torch.randn(84, 120, device="cuda"))
        self.fc2_bias   = nn.Parameter(torch.randn(84, device="cuda"))

        self.fc3_weight = nn.Parameter(torch.randn(num_classes, 84, device="cuda"))
        self.fc3_bias   = nn.Parameter(torch.randn(num_classes, device="cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv1 + ReLU + maxpool
        x = conv2d_relu(x, self.conv1_weight, stride=1)
        x = maxpool2d(x, stride=2)

        # conv2 + ReLU + maxpool
        x = conv2d_relu(x, self.conv2_weight, stride=1)
        x = maxpool2d(x, stride=2)

        # flatten
        x = x.view(x.size(0), -1)

        # fc1 + ReLU
        x = linear_relu(x, self.fc1_weight, self.fc1_bias)

        # fc2 + ReLU
        x = linear_relu(x, self.fc2_weight, self.fc2_bias)

        # fc3
        x = linear_relu(x, self.fc3_weight, self.fc3_bias)  # final linear + ReLU (optional)

        return x