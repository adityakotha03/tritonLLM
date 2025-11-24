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

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def fused_bn_bias_swish_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr, output_ptr,
    num_elements, size, eps,
    BLOCK_SIZE: tl.constexpr,
    DIV_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    col_offset = offsets % size
    mean = tl.load(running_mean_ptr + col_offset, mask=mask, other=0.0)
    inv_var = tl.load(running_var_ptr + col_offset, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + col_offset, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offset, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + 0)  # scalar bias

    x_hat = (x - mean) * tl.rsqrt(inv_var + eps)
    y = gamma * x_hat + beta + bias
    y = y / DIV_VAL
    sigmoid_y = tl.sigmoid(y)
    out = y * sigmoid_y

    tl.store(output_ptr + offsets, out, mask=mask)


def triton_matmul(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous()

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=None,
    )
    return c


def triton_fused_bn_bias_swish(x, weight, bias, running_mean, running_var, gamma, beta, eps, divide_value):
    assert x.is_cuda
    out = torch.empty_like(x)
    num_elements = x.numel()
    size = x.shape[-1]
    BLOCK_SIZE = 1024
    grid = lambda meta: ((num_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_bn_bias_swish_kernel[grid](
        x, weight, bias, running_mean, running_var,
        gamma, beta, out,
        num_elements, size, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        DIV_VAL=divide_value,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with Triton kernels for fused matmul and batch norm + bias + divide + swish.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_param = nn.Parameter(torch.randn(bias_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.divide_value = divide_value
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = triton_matmul(x, self.weight.t())
        x = x.to(torch.float32)
        x = triton_fused_bn_bias_swish(
            x, self.weight, self.bias_param,
            self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias,
            self.bn.eps, self.divide_value
        )
        return x