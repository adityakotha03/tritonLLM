import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel for matrix multiplication (gemm)
# Uses tensor cores (fp16) for 16×16 tiles
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_fp16_kernel(
    X_ptr,  # (M, K)
    W_ptr,  # (K, N)
    Y_ptr,  # (M, N)
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_Xm: tl.constexpr,
    stride_Xk: tl.constexpr,
    stride_Wk: tl.constexpr,
    stride_Wn: tl.constexpr,
    stride_Ym: tl.constexpr,
    stride_Yn: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Allocate accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_start = k

        # Load tiles of X and W
        X_block = tl.load(
            X_ptr + (m_start[:, None] * stride_Xm + k_start[None, :] * stride_Xk),
            mask=(m_start[:, None] < M) & (k_start[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        W_block = tl.load(
            W_ptr + (k_start[:, None] * stride_Wk + n_start[None, :] * stride_Wn),
            mask=(k_start[:, None] < K) & (n_start[None, :] < N),
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(X_block, W_block)

    # Store result
    tl.store(
        Y_ptr + (m_start[:, None] * stride_Ym + n_start[None, :] * stride_Yn),
        acc.to(tl.float16),
        mask=(m_start[:, None] < M) & (n_start[None, :] < N),
    )

def triton_matmul(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    X: (M, K)
    W: (K, N)
    """
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    Y = torch.empty((M, N), device=X.device, dtype=torch.float16)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_fp16_kernel[grid](
        X,
        W,
        Y,
        M,
        N,
        K,
        stride_Xm=X.stride(0),
        stride_Xk=X.stride(1),
        stride_Wk=W.stride(0),
        stride_Wn=W.stride(1),
        stride_Ym=Y.stride(0),
        stride_Yn=Y.stride(1),
    )
    return Y.to(torch.float32)

# ----------------------------------------------------------------------
# Triton kernel that fuses: max over dim, subtraction of mean, GELU
# ----------------------------------------------------------------------
@triton.jit
def fused_ops_kernel(
    X_ptr,            # (B, D)
    out_ptr,          # (B, D)
    B,
    D,
    dim,              # dim to reduce (0 or 1)
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    b_start = pid_b * BLOCK_B
    d_start = pid_d * BLOCK_D

    # Load a tile of X
    offs_b = b_start + tl.arange(0, BLOCK_B)[:, None]
    offs_d = d_start + tl.arange(0, BLOCK_D)[None, :]
    mask_b = offs_b < B
    mask_d = offs_d < D

    X_tile = tl.load(
        X_ptr + offs_b * X.stride(0) + offs_d * X.stride(1),
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Reduce along the specified dim
    if dim == 0:  # reduce over batch
        # Sum over rows
        sum_rows = tl.sum(X_tile, axis=0)
        max_vals = tl.max(X_tile, axis=0)
        mean_rows = sum_rows / B
    else:         # dim == 1, reduce over features
        sum_rows = tl.sum(X_tile, axis=1)
        max_vals = tl.max(X_tile, axis=1)
        mean_rows = sum_rows / D

    # Broadcast max and mean back to tile shape
    if dim == 0:
        max_broadcast = tl.broadcast_to(max_vals, X_tile.shape)
        mean_broadcast = tl.broadcast_to(mean_rows, X_tile.shape)
    else:
        max_broadcast = tl.broadcast_to(max_vals[:, None], X_tile.shape)
        mean_broadcast = tl.broadcast_to(mean_rows[:, None], X_tile.shape)

    # Subtract mean and apply GELU
    Y_tile = X_tile - mean_broadcast
    Y_tile = 0.5 * Y_tile * (1.0 + tl.math.erf(Y_tile / tl.sqrt(2.0)))

    # Store output
    tl.store(
        out_ptr + offs_b * out_ptr.stride(0) + offs_d * out_ptr.stride(1),
        Y_tile,
        mask=mask_b[:, None] & mask_d[None, :],
    )

def fused_ops(x: torch.Tensor, dim: int) -> torch.Tensor:
    B, D = x.shape
    out = torch.empty_like(x)

    BLOCK_B = 128
    BLOCK_D = 128

    grid = lambda meta: (
        (B + meta["BLOCK_B"] - 1) // meta["BLOCK_B"],
        (D + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
    )

    fused_ops_kernel[grid](
        x,
        out,
        B,
        D,
        dim,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )
    return out

# ----------------------------------------------------------------------
# Optimized model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a GEMM (fp16 tensor core), followed by a fused
    max/mean/GELU operation. The linear layer is implemented with a
    custom Triton matmul kernel.
    """
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        # Weight and bias stored as fp32 but will be cast to fp16 in kernel
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device='cuda'))
        self.bias = nn.Parameter(torch.randn(out_features, device='cuda'))
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # GEMM: X (B, in) @ W.T (in, out)  -> (B, out)
        matmul_out = triton_matmul(x, self.weight.t())
        # Add bias
        matmul_out = matmul_out + self.bias
        # Fuse max, mean, GELU
        out = fused_ops(matmul_out, self.max_dim)
        return out