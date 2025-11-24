import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_min_sub_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, constant,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_BIAS: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
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

    if USE_BIAS:
        bias_ptrs = b_ptr + offs_n * stride_wn
        bias_mask = offs_n < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        accumulator = accumulator + bias[None, :]

    out = accumulator.to(tl.float32)
    out = tl.minimum(out, constant)
    out = out - constant

    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + (offs_out_m[:, None] * stride_om + offs_out_n[None, :] * stride_on)
    out_mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)


def triton_matmul_min_sub(x, weight, bias, constant):
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32
    if bias is not None:
        assert bias.is_cuda and bias.dtype == torch.float32

    M, K = x.shape
    K, N = weight.shape

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    use_bias = bias is not None
    matmul_min_sub_kernel[grid](
        x, weight, bias, out, constant,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        GROUP_M=8,
        USE_BIAS=use_bias,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel for linear + min + subtract.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        return triton_matmul_min_sub(x, self.linear.weight, self.linear.bias, self.constant)