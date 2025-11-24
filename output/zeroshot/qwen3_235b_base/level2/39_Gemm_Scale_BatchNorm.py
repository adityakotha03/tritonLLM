import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_scale_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scale,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

    scale_ptrs = scale_ptr + offs_n * stride_scale
    scale_mask = offs_n < N
    scale = tl.load(scale_ptrs, mask=scale_mask)
    c_scaled = c * scale[None, :]
    tl.store(c_ptrs, c_scaled, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr,
    N, C,
    stride_xn, stride_xc,
    stride_yn, stride_yc,
    eps,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    x_ptrs = x_ptr + offs_n[:, None] * stride_xn + offs_c[None, :] * stride_xc
    mask = (offs_n[:, None] < N) & (offs_c[None, :] < C)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + offs_c, mask=offs_c < C, other=0.0)
    rstd = tl.load(rstd_ptr + offs_c, mask=offs_c < C, other=0.0)
    weight = tl.load(weight_ptr + offs_c, mask=offs_c < C, other=0.0)
    bias = tl.load(bias_ptr + offs_c, mask=offs_c < C, other=0.0)

    x_hat = (x - mean[None, :]) * rstd[None, :]
    y = x_hat * weight[None, :] + bias[None, :]

    y_ptrs = y_ptr + offs_n[:, None] * stride_yn + offs_c[None, :] * stride_yc
    tl.store(y_ptrs, y, mask=mask)


def triton_matmul_scale(x, weight, scale):
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) *
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    matmul_scale_kernel[grid](
        x, weight, y, scale,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        scale.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return y


def triton_batch_norm(x, weight, bias, running_mean, running_var, eps):
    N, C = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE_N = 64
    BLOCK_SIZE_C = 64

    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        triton.cdiv(C, META['BLOCK_SIZE_C']),
    )

    rstd = torch.rsqrt(running_var + eps)
    batch_norm_kernel[grid](
        x, y, weight, bias, running_mean, rstd,
        N, C,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernels for matmul+scale and custom batch norm.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = triton_matmul_scale(x, self.weight, self.scale)
        x = triton_batch_norm(
            x, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps
        )
        return x