import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    bias_ptr,
    stride_a,
    stride_b,
    stride_c,
    M,
    N,
    K,
    bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_a + offs_k[None, :],
                    mask=mask_m[:, None] & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_b + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & mask_n[None, :],
                    other=0.0)
        acc += tl.dot(a, b)

    if bias:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    tl.store(C_ptr + offs_m[:, None] * stride_c + offs_n[None, :],
             acc,
             mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    x: (batch, in_features)
    weight: (out_features, in_features)
    bias: (out_features,) or None
    """
    batch, in_f = x.shape
    out_f, _ = weight.shape
    out = torch.empty(batch, out_f, device=x.device, dtype=x.dtype)

    stride_x = x.stride(0)
    stride_w = weight.stride(0)
    stride_o = out.stride(0)

    grid = lambda meta: (
        triton.cdiv(batch, meta["BLOCK_SIZE_M"]),
        triton.cdiv(out_f, meta["BLOCK_SIZE_N"]),
    )

    _matmul_kernel[grid](
        x,
        weight,
        out,
        bias,
        stride_x,
        stride_w,
        stride_o,
        batch,
        out_f,
        in_f,
        bias is not None,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C"],
)
@triton.jit
def _avgpool_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offs < batch * channels
    b = offs // channels
    c = offs % channels

    # compute mean over spatial dims
    acc = tl.zeros([], dtype=tl.float32)
    for i in range(height):
        for j in range(width):
            idx = (b * stride_b +
                   c * stride_c +
                   i * stride_h +
                   j * stride_w)
            acc += tl.load(x_ptr + idx, mask=mask, other=0.0)
    acc /= (height * width)
    tl.store(out_ptr + offs, acc, mask=mask)


def triton_global_avgpool(x: torch.Tensor):
    """
    x: (batch, channels, height, width)
    returns (batch, channels)
    """
    batch, channels, height, width = x.shape
    out = torch.empty(batch, channels, device=x.device, dtype=x.dtype)

    stride_b = x.stride(0)
    stride_c = x.stride(1)
    stride_h = x.stride(2)
    stride_w = x.stride(3)

    grid = lambda meta: (triton.cdiv(batch * channels, meta["BLOCK_SIZE"]),)

    _avgpool_kernel[grid](
        x,
        out,
        stride_b,
        stride_c,
        stride_h,
        stride_w,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE=256,
    )
    return out

# ----------------------------------------------------------------------
# Optimised RegNet model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]

        self.feature_extractor = nn.Sequential(*layers)
        self.fc_weight = nn.Parameter(
            torch.empty(block_widths[-1], output_classes, device="cuda")
        )
        self.fc_bias = nn.Parameter(
            torch.empty(output_classes, device="cuda")
        )
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc_bias, -bound, bound)

    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = triton_global_avgpool(x)          # (B, C)
        x = triton_linear(x, self.fc_weight.t(), self.fc_bias)  # weight transpose for (C, out)
        return x