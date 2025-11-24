import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    # GELU activation
    sqrt_2_over_pi = 0.7978845608028654
    c_gelu = c * 0.5 * (1.0 + tl.tanh(sqrt_2_over_pi * c * (1.0 + 0.044715 * c * c)))

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_gelu, mask=c_mask)


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    n_rows, n_cols,
    stride_om, stride_on,
    stride_im, stride_in,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    row_ids = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_ids = tl.arange(0, BLOCK_SIZE_N)
    input_ptrs = input_ptr + row_ids[:, None] * stride_im + col_ids[None, :] * stride_in
    output_ptrs = output_ptr + row_ids[:, None] * stride_om + col_ids[None, :] * stride_on

    mask = (row_ids < n_rows)[:, None] & (col_ids < n_cols)[None, :]

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_matmul_gelu(x: torch.Tensor, weight: torch.Tensor):
    M, K = x.shape
    K, N = weight.shape

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    matmul_gelu_kernel[grid](
        x, weight, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return c


def triton_softmax(x: torch.Tensor, dim: int):
    if dim < 0:
        dim = x.ndim + dim
    n_rows = 1
    for i in range(dim):
        n_rows *= x.shape[i]
    n_cols = x.shape[dim]

    z = x.view(n_rows, n_cols)
    output = torch.empty_like(z)

    def grid(META):
        return (triton.cdiv(n_rows, META['BLOCK_SIZE_M']),)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128

    softmax_kernel[grid](
        output, z,
        n_rows, n_cols,
        output.stride(0), output.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return output.view_as(x)


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernels for matmul+GELU and custom softmax.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x @ self.weight.t() + self.bias
        x = triton_matmul_gelu(x, self.weight.t())
        x = triton_softmax(x, dim=1)
        return x