import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_swish_add_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    bias_stride,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator = tl.dot(x, w, acc=accumulator)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    b_ptrs = b_ptr + offs_n * bias_stride
    b_mask = offs_n < N
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    accumulator = accumulator + bias[None, :]

    sigmoid_out = tl.sigmoid(accumulator)
    swish_out = accumulator * sigmoid_out

    tl.store(out_ptrs, swish_out, mask=out_mask)


@triton.jit
def group_norm_kernel(
    x_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    y_ptr,
    M, N, G, H,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size_m = tl.cdiv(M, G)
    start_m = pid_m * BLOCK_M
    start_g = pid_g * group_size_m

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.sum(x, axis=1) / N
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    weight = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    y = (x - mean[:, None]) * rstd[:, None] * weight[None, :] + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask)

    tl.store(mean_ptr + pid_g * group_size_m + offs_m, mean, mask=offs_m < M)
    tl.store(rstd_ptr + pid_g * group_size_m + offs_m, rstd, mask=offs_m < M)


def triton_matmul_swish_add(x, w, b):
    M, K = x.shape
    K, N = w.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8

    matmul_swish_add_kernel[grid](
        x, w, b, out,
        b.stride(0),
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return out


def triton_group_norm(x, weight, bias, num_groups):
    M, N = x.shape
    G = num_groups
    H = N // G
    assert N % G == 0, "out_features must be divisible by num_groups"

    y = torch.empty_like(x)
    mean = torch.empty((G, triton.cdiv(M, G)), device=x.device, dtype=torch.float32)
    rstd = torch.empty((G, triton.cdiv(M, G)), device=x.device, dtype=torch.float32)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), G)

    BLOCK_M = 64
    BLOCK_N = 64

    group_norm_kernel[grid](
        x, mean, rstd, weight, bias,
        y,
        M, N, G, H,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for fused matmul+swish+bias and GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)
        self.num_groups = num_groups
        self.out_features = out_features
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = triton_matmul_swish_add(x, self.weight, self.bias)
        x = triton_group_norm(x, self.group_norm_weight, self.group_norm_bias, self.num_groups)
        return x