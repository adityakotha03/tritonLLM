import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# 1.  Fused GEMM + Bias + Hardtanh + Mish
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128},
                      num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128},
                      num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_gemm_bias_hardtanh_mish_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # Offsets for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Matrix multiplication with accumulation
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an,
                    mask=offs_m[:, None] < M,
                    other=0.0)

        b = tl.load(B_ptr + offs_k[:, None] * stride_bm + offs_n[None, :] * stride_bn,
                    mask=offs_n[None, :] < N,
                    other=0.0)

        acc += tl.dot(a, b)

    # Add bias, apply hardtanh and mish
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias

    # Hardtanh: clamp to [-6, 6]
    acc = tl.where(acc < -6.0, -6.0, acc)
    acc = tl.where(acc > 6.0, 6.0, acc)

    # Mish: x * tanh(softplus(x))
    softplus = tl.log1p(tl.exp(acc))
    mish = acc * tl.tanh(softplus)

    # Store result
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             mish,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_gemm_bias_hardtanh_mish(A: torch.Tensor,
                                 B: torch.Tensor,
                                 bias: torch.Tensor) -> torch.Tensor:
    """
    A : (M, K) float32/bfloat16
    B : (K, N) float32/bfloat16
    bias: (N,) float32
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # Make tensors contiguous
    A = A.contiguous()
    B = B.contiguous()
    bias = bias.contiguous()

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    _fused_gemm_bias_hardtanh_mish_kernel[grid](
        A,
        B,
        bias,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )

    return C


# -----------------------------
# 2.  Fused GroupNorm kernel
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_G": 64, "BLOCK_C": 64, "BLOCK_B": 64}, num_warps=4),
        triton.Config({"BLOCK_G": 128, "BLOCK_C": 128, "BLOCK_B": 64}, num_warps=8),
    ],
    key=["G", "C", "B"],
)
@triton.jit
def _groupnorm_kernel(
    x_ptr,  # (B, C)
    mean_ptr,  # (G,)
    var_ptr,   # (G,)
    gamma_ptr,  # (C,)
    beta_ptr,   # (C,)
    y_ptr,
    B, C, G,
    stride_b, stride_c,
    eps: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    g_start = pid_g * BLOCK_G
    c_start = pid_c * BLOCK_C
    b_start = pid_b * BLOCK_B

    offs_g = g_start + tl.arange(0, BLOCK_G)
    offs_c = c_start + tl.arange(0, BLOCK_C)
    offs_b = b_start + tl.arange(0, BLOCK_B)

    mask_g = offs_g < G
    mask_c = offs_c < C
    mask_b = offs_b < B

    # Load a tile of the input
    x_tile = tl.load(x_ptr + offs_b[:, None] * stride_b + offs_c[None, :] * stride_c,
                     mask=mask_b[:, None] & mask_c[None, :],
                     other=0.0)

    # Compute group id for each channel
    group_id = tl.arange(0, BLOCK_C) // (C // G)
    group_id = tl.broadcast_to(group_id[None, :], x_tile.shape)

    # Reduce sum and sum of squares per group
    sum_ = tl.sum(x_tile * tl.where(group_id == tl.arange(0, BLOCK_G)[:, None], 1.0, 0.0), 0)
    sum_sq = tl.sum(x_tile * x_tile * tl.where(group_id == tl.arange(0, BLOCK_G)[:, None], 1.0, 0.0), 0)

    # Write partial sums to shared memory (use global memory for simplicity)
    # Here we assume each group fits into a block, so we can write directly
    tl.store(mean_ptr + offs_g, sum_ / (B * (C // G)), mask=mask_g)
    tl.store(var_ptr + offs_g, sum_sq / (B * (C // G)) - (mean_ptr + offs_g) * (mean_ptr + offs_g), mask=mask_g)

    # Broadcast mean and var
    mean = tl.load(mean_ptr + offs_g, mask=mask_g, other=0.0)
    var = tl.load(var_ptr + offs_g, mask=mask_g, other=0.0)

    # Normalization
    inv_std = tl.math.rsqrt(var + eps)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)

    y = (x_tile - mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]

    tl.store(y_ptr + offs_b[:, None] * stride_b + offs_c[None, :] * stride_c,
             y,
             mask=mask_b[:, None] & mask_c[None, :])


def groupnorm(x: torch.Tensor,
              weight: torch.Tensor,
              bias: torch.Tensor,
              num_groups: int,
              eps: float = 1e-5) -> torch.Tensor:
    """
    x: (B, C) float32/bfloat16
    weight: (C,)
    bias: (C,)
    """
    B, C = x.shape
    G = num_groups
    assert C % G == 0

    mean = torch.empty(G, dtype=x.dtype, device=x.device)
    var = torch.empty(G, dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        (G + meta["BLOCK_G"] - 1) // meta["BLOCK_G"],
        (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        (B + meta["BLOCK_B"] - 1) // meta["BLOCK_B"],
    )

    _groupnorm_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        x,  # we reuse x buffer for output
        B,
        C,
        G,
        x.stride(0),
        x.stride(1),
        eps=eps,
        BLOCK_B=meta["BLOCK_B"],
        BLOCK_C=meta["BLOCK_C"],
        BLOCK_G=meta["BLOCK_G"],
    )
    return x


# -----------------------------
# 3.  Optimized Model
# -----------------------------
class ModelNew(nn.Module):
    """
    Optimized model that replaces the GEMM, bias add, hardtanh, mish,
    and groupnorm with custom Triton kernels for higher throughput.
    """

    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Linear weights are stored as bf16 for Tensor Core
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device='cuda', dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.randn(bias_shape, device='cuda', dtype=torch.bfloat16))

        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(out_features, device='cuda', dtype=torch.bfloat16))
        self.gn_bias = nn.Parameter(torch.zeros(out_features, device='cuda', dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are bf16
        x = x.to(torch.bfloat16)

        # GEMM + bias + hardtanh + mish
        y = fused_gemm_bias_hardtanh_mish(x, self.weight.t(), self.bias)

        # GroupNorm
        y = groupnorm(y, self.gn_weight, self.gn_bias, self.num_groups)

        # Convert back to float32 if needed
        return y.to(torch.float32)