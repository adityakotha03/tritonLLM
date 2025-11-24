import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Triton kernel that fuses:  A @ Wᵀ  + bias   →   Swish(A @ Wᵀ + bias)
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_swish_bias_fused(
    A_ptr,          # (M, K)
    W_ptr,          # (N, K)
    bias_ptr,       # (N,)
    out_ptr,        # (M, N)
    M, N, K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute the start indices for this tile
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k

        # Load tiles of A and W
        A_tile = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        W_tile = tl.load(
            W_ptr + offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(A_tile, W_tile, precision=tl.float32)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Swish: x * sigmoid(x)
    acc_swish = acc * tl.sigmoid(acc)

    # Store result
    tl.store(
        out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc_swish,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_matmul_swish_bias(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    a: (batch, in_features)
    weight: (out_features, in_features)   -- stored in row-major (for matmul A @ Wᵀ)
    bias: (out_features,)
    """
    assert a.is_cuda and weight.is_cuda and bias.is_cuda

    M, K = a.shape
    N = weight.shape[0]

    # Output tensor
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    matmul_swish_bias_fused[grid](
        a,
        weight,
        bias,
        out,
        M, N, K,
        a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return out


# --------------------------------------------------------------------
# Optimized model using the fused Triton kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that uses a fused Triton kernel for:
      1) Linear transformation (matmul)
      2) Swish activation
      3) Bias addition
    GroupNorm is still performed with PyTorch for simplicity.
    """

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias_shape: tuple):
        super().__init__()
        self.linear_weight = nn.Parameter(torch.randn(out_features, in_features, device="cuda"))
        self.bias = nn.Parameter(torch.randn(bias_shape, device="cuda"))
        self.group_norm = nn.GroupNorm(num_groups, out_features, eps=1e-5, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()
        # Fused matmul + swish + bias
        x = fused_matmul_swish_bias(x, self.linear_weight, self.bias)
        # GroupNorm
        x = self.group_norm(x)
        return x