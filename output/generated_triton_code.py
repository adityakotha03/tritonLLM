import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_leakyrelu_kernel(
    A_ptr,  # (M, K)
    W_ptr,  # (N, K) - will be accessed as (K, N) logically
    B_ptr,  # (N,) bias
    Out_ptr,  # (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_b,
    stride_om,
    stride_on,
    multiplier,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k_init = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k_init[None, :] * stride_ak
    w_ptrs = W_ptr + offs_k_init[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_mask = (offs_m[:, None] < M) & (k + offs_k_init[None, :] < K)
        w_mask = (k + offs_k_init[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, w)

        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Epilogue: add bias, scale, leaky ReLU
    bias = tl.load(B_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    acc = acc * multiplier
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store result
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_linear_leakyrelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, multiplier: float, negative_slope: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16/bf16/fp32"
    assert weight.dtype == x.dtype and bias.dtype == x.dtype, "x, weight, and bias must have the same dtype"

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "Incompatible dimensions for matmul."

    # Ensure contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)  # accumulate/store in fp32 for stability

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    fused_linear_leakyrelu_kernel[grid](
        x_c, w_c, b_c, out,
        M, N, K,
        x_c.stride(0), x_c.stride(1),
        w_c.stride(0), w_c.stride(1),
        b_c.stride(0),
        out.stride(0), out.stride(1),
        multiplier, negative_slope,
    )

    # Cast back to original dtype to match PyTorch Linear behavior
    return out.to(x.dtype)


class ModelNew(nn.Module):
    """
    Fused Linear (GEMM) + scale + LeakyReLU implemented with a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        return triton_linear_leakyrelu(x, self.weight, self.bias, self.multiplier, self.negative_slope)


# Parameters and input helpers consistent with the original
batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]