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
    GROUP_SIZE_M: tl.constexpr
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
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, sigmoid, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row_start_x = pid_m * stride_xm
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(x_ptr + row_start_x + offs_n, mask=offs_n < N, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    logsumexp = x_max + tl.log(sum_exp)
    out_offset = pid_m * stride_om
    tl.store(out_ptr + out_offset, logsumexp)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    if M * N == 0:
        return c

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return c


def triton_sigmoid(x: torch.Tensor):
    assert x.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def triton_logsumexp(x: torch.Tensor, dim: int):
    assert x.is_cuda
    out_shape = list(x.shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    M, N = x.shape
    assert dim == 1
    grid = lambda meta: (M,)
    logsumexp_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0),
        BLOCK_SIZE_N=1024,
    )
    return out.squeeze(dim)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1_weight = nn.Parameter(torch.randn(hidden_size, input_size).cuda())
        self.linear1_bias = nn.Parameter(torch.randn(hidden_size).cuda())
        self.linear2_weight = nn.Parameter(torch.randn(output_size, hidden_size).cuda())
        self.linear2_bias = nn.Parameter(torch.randn(output_size).cuda())

    def forward(self, x):
        x = x.to(torch.float32)
        w1 = self.linear1_weight.to(torch.float32)
        b1 = self.linear1_bias.to(torch.float32)
        w2 = self.linear2_weight.to(torch.float32)
        b2 = self.linear2_bias.to(torch.float32)

        x = triton_matmul(x, w1.t())
        x = x + b1
        x = triton_sigmoid(x)
        x = triton_matmul(x, w2.t())
        x = x + b2
        x = triton_logsumexp(x, dim=1)
        return x