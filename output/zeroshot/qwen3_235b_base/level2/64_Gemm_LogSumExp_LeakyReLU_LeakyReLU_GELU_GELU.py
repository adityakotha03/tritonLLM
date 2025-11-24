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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am < M)[:, None] & (offs_k < K)[None, :]
        b_mask = (offs_k < K)[:, None] & (offs_bn < N)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    if ACTIVATION == "leaky_relu":
        c = c + 0.01 * tl.where(c >= 0, c, c)
    elif ACTIVATION == "gelu":
        c = 0.5 * c * (1.0 + tl.math.erf(c * 0.70710678))
    c = tl.where(c_mask, c, 0.0)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def logsumexp_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_input_m, stride_input_n,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    input_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_n[None, :] * stride_input_n
    input = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(input, axis=1)
    row_max = tl.where(row_max == -float('inf'), 0.0, row_max)
    input_minus_max = input - row_max[:, None]
    exp_input = tl.exp(input_minus_max)
    sum_exp = tl.sum(exp_input, axis=1)
    logsumexp = row_max + tl.log(sum_exp)

    output_ptrs = output_ptr + offs_m * stride_output_m + pid_n * stride_output_n
    output_mask = (offs_m < M)
    tl.store(output_ptrs, logsumexp, mask=output_mask)


@triton.jit
def leaky_relu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    negative_slope: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x >= 0, x, negative_slope * x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = 0.5 * x * (1.0 + tl.math.erf(x * 0.70710678))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_matmul_gemm(a, b, activation=None):
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
        ACTIVATION=activation,
    )
    return c


def triton_logsumexp(x, dim):
    assert x.is_cuda
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    logsumexp_kernel[grid](
        x,
        output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output


def triton_leaky_relu(x, negative_slope=0.01):
    assert x.is_cuda
    x = x.contiguous()
    n_elements = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    leaky_relu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, negative_slope=negative_slope)
    return y


def triton_gelu(x):
    assert x.is_cuda
    x = x.contiguous()
    n_elements = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_val = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_val', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias_val, -bound, bound)

    def forward(self, x):
        # Gemm: x @ weight.T + bias
        weight_T = self.weight.t()
        x = triton_matmul_gemm(x, weight_T)
        if self.bias_val is not None:
            x = x + self.bias_val.unsqueeze(0)
        # LogSumExp
        x = triton_logsumexp(x, dim=1)
        # LeakyReLU x2
        x = triton_leaky_relu(x, negative_slope=0.01)
        x = triton_leaky_relu(x, negative_slope=0.01)
        # GELU x2
        x = triton_gelu(x)
        x = triton_gelu(x)
        return x