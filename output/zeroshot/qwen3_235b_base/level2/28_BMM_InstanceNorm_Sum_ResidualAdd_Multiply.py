import torch
import torch.nn as nn
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    if ACTIVATION == "bias":
        bias_ptr = c_ptr + offs_bn
        bias = tl.load(bias_ptr)
        c += bias[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def instance_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start_m = pid_m * BLOCK_SIZE_M
    col_start_n = pid_n * BLOCK_SIZE_N
    row_end_m = min(row_start_m + BLOCK_SIZE_M, M)
    col_end_n = min(col_start_n + BLOCK_SIZE_N, N)

    mask = (row_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
    mask &= (col_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N

    offsets = (row_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_xm + \
              (col_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_xn
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=1) / N
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / N
    inv_var = tl.rsqrt(var + eps)
    normed = diff * inv_var[:, None]

    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + (col_start_n + tl.arange(0, BLOCK_SIZE_N)), mask=col_start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=1.0)
        normed = normed * gamma[None, :]
    if beta_ptr is not None:
        beta = tl.load(beta_ptr + (col_start_n + tl.arange(0, BLOCK_SIZE_N)), mask=col_start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
        normed = normed + beta[None, :]

    tl.store(y_ptr + offsets, normed, mask=mask)


@triton.jit
def fused_add_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    res = (x + y) * y
    tl.store(out_ptr + offsets, res, mask=mask)


def triton_matmul_bias(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        x, weight, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION="bias",
    )
    return y


def triton_instance_norm(x, gamma, beta, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    instance_norm_kernel[grid](
        x, gamma, beta, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32,
    )
    return y


def triton_fused_add_mul(x, y):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_add_mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for fused operations.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.eps = eps

    def forward(self, x, y):
        x = triton_matmul_bias(x, self.weight, self.bias)
        x = x.unsqueeze(1).unsqueeze(1)
        x = triton_instance_norm(x, self.gamma, self.beta, self.eps)
        x = x.squeeze(1).squeeze(1)
        x = triton_fused_add_mul(x, y)
        return x