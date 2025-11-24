import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    if ACTIVATION == "bias":
        bias_ptrs = b_ptr + offs_n * stride_bn
        bias = tl.load(bias_ptrs)
        c += bias[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(x, weight, bias=None):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on GPU."
    x = x.contiguous()
    weight = weight.t().contiguous()  # Transpose weight for efficient access
    M, K = x.shape
    K, N = weight.shape

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    if bias is not None:
        bias = bias.contiguous()

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    matmul_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION="bias" if bias is not None else "none"
    )
    return out


@triton.jit
def batch_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, running_mean_ptr, running_var_ptr,
    y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)

    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.load(running_mean_ptr + cols, mask=cols < N, other=0.0)
    var = tl.load(running_var_ptr + cols, mask=cols < N, other=0.0)
    gamma = tl.load(gamma_ptr + cols, mask=cols < N, other=1.0)
    beta = tl.load(beta_ptr + cols, mask=cols < N, other=0.0)

    x_hat = (x - mean[None, :]) * tl.rsqrt(var[None, :] + eps)
    y = gamma[None, :] * x_hat + beta[None, :]

    y_ptrs = y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask)


def triton_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5):
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Inputs must be on GPU."
    x = x.contiguous()
    M, N = x.shape

    out = torch.empty_like(x)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128

    batch_norm_kernel[grid](
        x, gamma, beta, running_mean, running_var,
        out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


@triton.jit
def scale_kernel(
    x_ptr, scale_ptr,
    out_ptr,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE_N
    offsets = start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(x, scale):
    assert x.is_cuda and scale.is_cuda, "Inputs must be on GPU."
    x = x.contiguous()
    N = x.numel()
    out = torch.empty_like(x)

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_N = 1024

    scale_kernel[grid](x, scale, out, N, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row_ptr = x_ptr + row_idx * stride_x_row
    x = tl.load(x_row_ptr + col_offsets * stride_x_col, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    softmax_output = num / den

    out_row_ptr = out_ptr + row_idx * stride_out_row
    tl.store(out_row_ptr + col_offsets * stride_out_col, softmax_output, mask=mask)


def triton_softmax(x):
    assert x.is_cuda, "Input must be on GPU."
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x, dtype=torch.float32)

    def grid(META):
        return (n_rows,)

    BLOCK_SIZE = 1024
    softmax_kernel[grid](x, out, n_rows, n_cols, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Gemm, BatchNorm, Scale, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn_eps = bn_eps

    def forward(self, x):
        x = triton_matmul(x, self.gemm.weight, self.gemm.bias)
        x = triton_batch_norm(x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn_eps)
        x = triton_scale(x, self.scale)
        x = triton_softmax(x)
        return x