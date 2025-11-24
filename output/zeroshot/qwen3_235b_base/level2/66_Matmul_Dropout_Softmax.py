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
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    if ACTIVATION == "bias":
        bias_ptr = b_ptr
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        c += bias[None, :]
    elif ACTIVATION == "bias_relu":
        bias_ptr = b_ptr
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        c += bias[None, :]
        c = tl.maximum(c, 0.0)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def dropout_kernel(
    x_ptr, output_ptr, noise_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.load(noise_ptr + offsets, mask=mask)
    prob = p
    scale = 1.0 / (1.0 - prob)
    keep = random > prob
    output = tl.where(keep, x * scale, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def softmax_kernel(
    x_ptr, output_ptr,
    n_rows, n_cols,
    stride_x_row, stride_x_col,
    stride_o_row, stride_o_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_row_ptr = x_ptr + row * stride_x_row
    x = tl.load(x_row_ptr + col_offsets * stride_x_col, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum
    output_row_ptr = output_ptr + row * stride_o_row
    tl.store(output_row_ptr + col_offsets * stride_o_col, x_softmax, mask=mask)


def triton_matmul_bias(x, weight, bias):
    M, K = x.shape
    N, K = weight.shape
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    def grid(meta): return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        x, weight, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION="bias",
    )
    return c


def triton_dropout(x, p, noise):
    n_elements = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    dropout_kernel[grid](x, y, noise, n_elements, p, BLOCK_SIZE=BLOCK_SIZE)
    return y


def triton_softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (M,)
    softmax_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout_p = dropout_p

    def forward(self, x):
        x = triton_matmul_bias(x, self.weight, self.bias)
        noise = torch.rand_like(x)
        x = triton_dropout(x, self.dropout_p, noise)
        x = triton_softmax(x)
        return x