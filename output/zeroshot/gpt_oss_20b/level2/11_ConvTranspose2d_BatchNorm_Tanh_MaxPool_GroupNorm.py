import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------- Triton kernels --------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_X": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_X": 512}, num_warps=8),
    ],
    key=["N", "C", "H", "W"],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr,          # input tensor (N, C, H, W)
    Y_ptr,          # output tensor (N, C, H/2, W/2)
    N, C, H, W,    # dimensions
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr = 4,  # pooling kernel height
    BLOCK_SIZE_Z: tl.constexpr = 4,  # pooling kernel width
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    n = pid_x
    c = pid_y
    h_out = pid_z // (W // 2)
    w_out = pid_z % (W // 2)

    h_start = h_out * 2
    w_start = w_out * 2

    max_val = -1e9
    for dh in range(BLOCK_SIZE_Y):
        for dw in range(BLOCK_SIZE_Z):
            h = h_start + dh
            w = w_start + dw
            if h < H and w < W:
                offset = ((n * C + c) * H + h) * W + w
                val = tl.load(X_ptr + offset)
                max_val = tl.maximum(max_val, val)

    out_offset = ((n * C + c) * (H // 2) + h_out) * (W // 2) + w_out
    tl.store(Y_ptr + out_offset, max_val)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_X": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_X": 512}, num_warps=8),
    ],
    key=["N", "C", "H", "W", "G"],
)
@triton.jit
def groupnorm_kernel(
    X_ptr,          # input tensor (N, C, H, W)
    Y_ptr,          # output tensor (N, C, H, W)
    N, C, H, W, G,  # dimensions
    eps: tl.constexpr = 1e-5,
    BLOCK_SIZE_X: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // ((C // G) * (H * W))
    rem = pid % ((C // G) * (H * W))
    g = rem // (H * W)
    c_in_group = rem % (H * W)

    # compute mean
    sum_val = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    for i in range(BLOCK_SIZE_X):
        idx = i
        if idx < (C // G) * H * W:
            c = g * (C // G) + idx // (H * W)
            h = (idx // W) % H
            w = idx % W
            offset = ((n * C + c) * H + h) * W + w
            sum_val[i] = tl.load(X_ptr + offset)
    mean = tl.sum(sum_val) / (C // G * H * W)

    # compute variance
    var = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    for i in range(BLOCK_SIZE_X):
        idx = i
        if idx < (C // G) * H * W:
            c = g * (C // G) + idx // (H * W)
            h = (idx // W) % H
            w = idx % W
            offset = ((n * C + c) * H + h) * W + w
            diff = tl.load(X_ptr + offset) - mean
            var[i] = diff * diff
    var = tl.sum(var) / (C // G * H * W)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # write normalized output
    for i in range(BLOCK_SIZE_X):
        idx = i
        if idx < (C // G) * H * W:
            c = g * (C // G) + idx // (H * W)
            h = (idx // W) % H
            w = idx % W
            offset = ((n * C + c) * H + h) * W + w
            val = tl.load(X_ptr + offset)
            norm = (val - mean) * inv_std
            tl.store(Y_ptr + offset, norm)

# -------------------- Model definition --------------------------------

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation,
    max pooling (via Triton), and group normalization (via Triton).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = self._max_pool
        self.group_norm = self._group_norm
        self.out_channels = out_channels
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

    # ------------------------------------------------------------------
    # Triton-backed max pooling
    # ------------------------------------------------------------------
    def _max_pool(self, x):
        N, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and width must be even for 2x2 pooling."
        out = torch.empty((N, C, H // 2, W // 2), dtype=x.dtype, device=x.device)
        grid = lambda meta: (N, C, (H // 2) * (W // 2))
        maxpool2d_kernel[grid](x, out, N, C, H, W)
        return out

    # ------------------------------------------------------------------
    # Triton-backed group normalization
    # ------------------------------------------------------------------
    def _group_norm(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        assert C % G == 0, "Number of channels must be divisible by num_groups."
        out = torch.empty_like(x)
        grid = lambda meta: (N * G * (C // G) * (H * W))
        groupnorm_kernel[grid](x, out, N, C, H, W, G)
        return out