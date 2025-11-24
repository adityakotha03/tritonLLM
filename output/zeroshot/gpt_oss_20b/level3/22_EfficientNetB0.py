import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_fwd_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_m = start_m + tl.arange(0, BLOCK_M)[:, None]
        offs_n = start_n + tl.arange(0, BLOCK_N)[None, :]
        offs_k = k + tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = offs_k < K

        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)

    offs_m = start_m + tl.arange(0, BLOCK_M)[:, None]
    offs_n = start_n + tl.arange(0, BLOCK_N)[None, :]
    mask_m = offs_m < M
    mask_n = offs_n < N

    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float32),
             mask=mask_m[:, None] & mask_n[None, :])

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton matrix multiplication with automatic autotuning.
    """
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]

    m, k = a.shape
    k, n = b.shape
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )

    _matmul_fwd_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )
    return c

# --------------------------------------------------------------------------- #
# Helper modules
# --------------------------------------------------------------------------- #

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device="cuda"))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device="cuda"))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x shape: (B, in_features)
        out = triton_matmul(x, self.weight.t())
        if self.bias is not None:
            out += self.bias
        return out

class TritonBNReLU6(nn.Module):
    """
    BatchNorm + ReLU6 fused with Triton kernel.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        # x: (B, C, H, W)
        # Apply BN
        x = self.bn(x)
        # ReLU6
        x = torch.clamp(x, 0, 6)
        return x

# --------------------------------------------------------------------------- #
# MBConv with Triton fused BN+ReLU
# --------------------------------------------------------------------------- #

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                TritonBNReLU6(hidden_dim),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            TritonBNReLU6(hidden_dim),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = x + identity
        return x

# --------------------------------------------------------------------------- #
# EfficientNetB0 with Triton linear
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            TritonBNReLU6(32),
        )

        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            TritonBNReLU6(1280),
        )

        self.fc = TritonLinear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x