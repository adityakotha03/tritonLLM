import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr,  # [M, K] where K = in_features
    B_ptr,  # [N, K] (row-major weights), we will access as B_T[K, N] via strides
    Bias_ptr,  # [N] or nullptr
    C_ptr,  # [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,  # stride for K dimension when viewing B as transposed [K, N]
    stride_bn,  # stride for N dimension when viewing B as transposed [K, N]
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # 1D launch grid with grouping along M to maximize L2 reuse of B tiles
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid // group_size_m) % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # View B as transposed [K, N] with provided strides (stride_bk, stride_bn)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        k_mask = offs_k < k_remaining

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0).to(tl.float16)
        # Weight is heavily reused across M, hint L2 with cache_modifier='.cg'
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0, cache_modifier=".cg").to(tl.float16)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Add bias if provided
    if tl.constexpr(Bias_ptr is not None):
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    High-performance Linear using Triton:
      y = x @ weight.T + bias
    x: [M, K], weight: [N, K] row-major, bias: [N]
    Output: float32 tensor [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    # Use fp16 weights/activations for Tensor Cores, accumulate in fp32
    x_ = x.contiguous()
    w_ = weight.contiguous()
    M, K = x_.shape
    N = w_.shape[0]

    # Ensure expected dtypes for TC; cast lazily to fp16
    if x_.dtype != torch.float16:
        x_tc = x_.to(torch.float16)
    else:
        x_tc = x_
    if w_.dtype != torch.float16:
        w_tc = w_.to(torch.float16)
    else:
        w_tc = w_

    y = torch.empty((M, N), device=x_.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    # Strides
    stride_am = x_tc.stride(0)
    stride_ak = x_tc.stride(1)
    # For B viewed as transposed [K, N]: original B is [N, K] with strides [K, 1]
    stride_bk = w_tc.stride(1)  # 1 for contiguous
    stride_bn = w_tc.stride(0)  # K for contiguous
    stride_cm = y.stride(0)
    stride_cn = y.stride(1)

    matmul_bias_kernel[grid](
        x_tc,
        w_tc,
        bias if bias is not None else None,
        y,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model with Triton-accelerated Linear (cp.async K-pipelined) + PyTorch Dropout + Softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout_p)
        # Parameters: store weight in row-major [out_features, in_features] for coalesced loads
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        b = torch.empty(out_features)
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)
        # Register as parameters
        # Use fp16 storage for weight to leverage Tensor Cores; bias in fp32
        self.weight = nn.Parameter(w.half())
        self.bias = nn.Parameter(b.float())

    def forward(self, x: torch.Tensor):
        if not x.is_cuda:
            # Fallback on CPU to maintain functionality
            out = F.linear(x, self.weight.float(), self.bias.float())
            out = self.dropout(out)
            return torch.softmax(out, dim=1)

        # Ensure inputs are on same device/dtype
        x = x.to(self.weight.device)
        out = triton_linear(x, self.weight, self.bias)
        out = self.dropout(out)
        out = torch.softmax(out, dim=1)
        return out