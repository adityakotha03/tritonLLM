import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------- Triton kernel for linear layer ----------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def linear_fused_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    out_ptr,
    M: tl.constexpr,  # batch size * hidden_dim
    N: tl.constexpr,  # out_dim
    K: tl.constexpr,  # in_dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        X = tl.load(
            X_ptr + (block_start_m + tl.arange(0, BLOCK_M)).to(tl.int64)[:, None] * K
            + (k + tl.arange(0, BLOCK_K))[None, :],
            mask=(block_start_m + tl.arange(0, BLOCK_M)) < M,
            other=0.0,
        )
        W = tl.load(
            W_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N
            + (block_start_n + tl.arange(0, BLOCK_N))[None, :],
            mask=(block_start_n + tl.arange(0, BLOCK_N)) < N,
            other=0.0,
        )
        acc += tl.dot(X, W)

    if pid_m == 0:
        B = tl.load(B_ptr + (block_start_n + tl.arange(0, BLOCK_N)), mask=(block_start_n + tl.arange(0, BLOCK_N)) < N, other=0.0)
        acc += B[None, :]

    tl.store(
        out_ptr + (block_start_m + tl.arange(0, BLOCK_M))[:, None] * N
        + (block_start_n + tl.arange(0, BLOCK_N))[None, :],
        acc,
        mask=(block_start_m + tl.arange(0, BLOCK_M)) < M
        & (block_start_n + tl.arange(0, BLOCK_N)) < N,
    )

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    x: (batch, in_dim)
    weight: (out_dim, in_dim)
    bias: (out_dim)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_fused_kernel[grid](
        x,
        weight.t(),          # weight is (out_dim, in_dim) -> we need (in_dim, out_dim)
        bias,
        out,
        M=M,
        N=N,
        K=K,
        BLOCK_M=meta['BLOCK_M'],
        BLOCK_N=meta['BLOCK_N'],
        BLOCK_K=meta['BLOCK_K'],
    )
    return out

# ---------- MobileNetV2 with Triton linear ----------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup
            layers = []
            if expand_ratio != 1:
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            return nn.Sequential(*layers), use_res_connect

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                blk, _ = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                features.append(blk)
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Linear(last_channel, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # replace linear with Triton kernel
        out = triton_linear(x, self.classifier.weight, self.classifier.bias)
        return out