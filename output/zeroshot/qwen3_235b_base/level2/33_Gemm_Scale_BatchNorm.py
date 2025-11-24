import torch
import torch.nn as nn
import torch.nn.functional as F
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float32)

    scale = tl.load(scale_ptr + offs_n * stride_scale, mask=offs_n < N, other=1.0).to(tl.float32)
    c = c * scale[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def batch_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    mean = tl.load(mean_ptr + offs_m * stride_xm, mask=mask_m, other=0.0)
    var = tl.load(var_ptr + offs_m * stride_xm, mask=mask_m, other=0.0)
    inv_var = tl.math.rsqrt(var + eps)

    offs_n = tl.arange(0, BLOCK_N)
    for off_n in range(0, N, BLOCK_N):
        offs_n_curr = off_n + offs_n
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n_curr[None, :] * stride_xn
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n_curr[None, :] * stride_yn
        mask = (offs_m[:, None] < M) & (offs_n_curr[None, :] < N)
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        bn = (x - mean[:, None]) * inv_var[:, None]
        weight = tl.load(weight_ptr + offs_n_curr * 1, mask=offs_n_curr < N, other=1.0)
        bias = tl.load(bias_ptr + offs_n_curr * 1, mask=offs_n_curr < N, other=0.0)
        y = bn * weight[None, :] + bias[None, :]
        tl.store(y_ptrs, y, mask=mask)


def triton_matmul_scale(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_M = 8

    matmul_scale_kernel[grid](
        x, weight, y, scale,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        scale.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M
    )
    return y


def triton_batch_norm(x: torch.Tensor, bn: nn.BatchNorm1d):
    M, N = x.shape
    y = torch.empty_like(x)

    weight = bn.weight
    bias = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']),)

    BLOCK_M, BLOCK_N = 64, 64

    batch_norm_kernel[grid](
        x, y, weight, bias, running_mean, running_var,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        bn.eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return y


class ModelNew(nn.Module):
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
        x = x + self.bias.unsqueeze(0)
        x = triton_batch_norm(x, self.bn)
        return x