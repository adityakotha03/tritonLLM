import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------- Triton kernel ----------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=16),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_bn_gelu_relu_kernel(
    X_ptr,          # (M, K)
    W_ptr,          # (K, N)
    B_ptr,          # (N,)
    BN_W_ptr,       # (N,)
    BN_B_ptr,       # (N,)
    RUNNING_MEAN_ptr,   # (N,)
    RUNNING_VAR_ptr,    # (N,)
    OUT_ptr,        # (M, N)
    M: tl.constexpr,  # batch size
    N: tl.constexpr,  # out_features
    K: tl.constexpr,  # in_features
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)

        a = tl.load(X_ptr + m_offsets[:, None] * K + k_offsets[None, :], mask=mask_m[:, None] & (k_offsets < K), other=0.0)
        b = tl.load(W_ptr + k_offsets[:, None] * N + n_offsets[None, :], mask=(k_offsets[:, None] < K) & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(B_ptr + n_offsets, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # BatchNorm
    mean = tl.load(RUNNING_MEAN_ptr + n_offsets, mask=mask_n, other=0.0)
    var  = tl.load(RUNNING_VAR_ptr  + n_offsets, mask=mask_n, other=0.0)

    std = tl.math.rsqrt(var + EPS)
    bn_w = tl.load(BN_W_ptr + n_offsets, mask=mask_n, other=0.0)
    bn_b = tl.load(BN_B_ptr + n_offsets, mask=mask_n, other=0.0)

    acc = (acc - mean[None, :]) * std[None, :] * bn_w[None, :] + bn_b[None, :]

    # GELU + ReLU
    acc = 0.5 * acc * (1.0 + tl.math.tanh(tl.math.sqrt(2.0 / tl.math.pi) * (acc + 0.044715 * tl.math.pow(acc, 3))))
    acc = tl.max(acc, 0.0)

    # Store result
    tl.store(OUT_ptr + m_offsets[:, None] * N + n_offsets[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_matmul_bn_gelu_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
):
    """
    x: (M, K)          input
    weight: (K, N)     linear weight
    bias: (N,)         linear bias
    bn_weight: (N,)    batchnorm weight (γ)
    bn_bias: (N,)      batchnorm bias  (β)
    running_mean: (N,)
    running_var: (N,)
    """
    assert x.is_cuda and weight.is_cuda
    M, K = x.shape
    K2, N = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        ( (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
          (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'] ),
    )

    fused_matmul_bn_gelu_relu_kernel[grid](
        x, weight, bias, bn_weight, bn_bias,
        running_mean, running_var, out,
        M, N, K, eps,
        BLOCK_SIZE_M=meta['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=meta['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=meta['BLOCK_SIZE_K'],
    )
    return out


# ---------- Model ----------
class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, BatchNorm, GELU and ReLU into one Triton kernel.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features, affine=True, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear part
        # Note: nn.Linear uses weight shape (out_features, in_features)
        # Triton expects (in_features, out_features)
        w = self.linear.weight.t().contiguous()          # (in_features, out_features)
        b = self.linear.bias.contiguous()                # (out_features,)

        # BatchNorm parameters
        bn_w = self.bn.weight.contiguous()                # (out_features,)
        bn_b = self.bn.bias.contiguous()                  # (out_features,)
        running_mean = self.bn.running_mean.contiguous()  # (out_features,)
        running_var = self.bn.running_var.contiguous()    # (out_features,)

        # Fuse everything
        out = fused_matmul_bn_gelu_relu(
            x,
            w,
            b,
            bn_w,
            bn_b,
            running_mean,
            running_var,
            eps=self.bn.eps,
        )
        return out