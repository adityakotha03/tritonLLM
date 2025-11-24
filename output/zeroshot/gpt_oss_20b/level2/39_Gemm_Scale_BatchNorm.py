import torch
import torch.nn as nn
import triton
import triton.language as tl
from functools import partial


# ------------------ Triton kernel for fused GEMM + element‑wise scaling ------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_scaled_kernel(
    a_ptr,          # pointer to input matrix A (M x K)
    w_ptr,          # pointer to weight matrix W (N x K) (stored as K x N in memory)
    scale_ptr,      # pointer to scale vector (N,)
    out_ptr,        # pointer to output matrix (M x N)
    M: tl.constexpr,   # number of rows in A / out
    N: tl.constexpr,   # number of columns in W / out
    K: tl.constexpr,   # inner dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_am: tl.constexpr,   # stride of A in rows
    stride_ak: tl.constexpr,   # stride of A in cols
    stride_wk: tl.constexpr,   # stride of W in rows (K)
    stride_wn: tl.constexpr,   # stride of W in cols (N)
    stride_outm: tl.constexpr, # stride of out in rows
    stride_outn: tl.constexpr, # stride of out in cols
    eps: tl.constexpr,         # small epsilon for safety
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Each program handles a tile of size BLOCK_M x BLOCK_N
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Allocate accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Load tiles of A and W
        a = tl.load(
            a_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_am
            + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
            mask=(row_start + tl.arange(0, BLOCK_M)[:, None] < M)
            & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )

        w = tl.load(
            w_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_wk
            + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_wn,
            mask=(k + tl.arange(0, BLOCK_K)[:, None] < K)
            & (col_start + tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        )

        # Accumulate
        acc += tl.dot(a, w)

    # Scale by the scale vector
    scale = tl.load(scale_ptr + (col_start + tl.arange(0, BLOCK_N)), mask=(col_start + tl.arange(0, BLOCK_N) < N), other=0.0)
    acc *= scale[None, :]

    # Store result
    tl.store(
        out_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_outm
        + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_outn,
        acc,
        mask=(row_start + tl.arange(0, BLOCK_M)[:, None] < M)
        & (col_start + tl.arange(0, BLOCK_N)[None, :] < N),
    )


def triton_gemm_scaled(a: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Perform (a @ w.t()) * scale element‑wise with Triton.
    a : (B, K)  float32
    w : (N, K)  float32   (weight matrix of nn.Linear)
    scale : (N,)   float32
    Returns:
        out : (B, N)
    """
    B, K = a.shape
    N, _ = w.shape

    out = torch.empty((B, N), dtype=a.dtype, device=a.device)

    # Grid: (num_blocks_m, num_blocks_n)
    grid = lambda meta: (
        (B + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    gemm_scaled_kernel[grid](
        a,
        w,
        scale,
        out,
        M=B,
        N=N,
        K=K,
        stride_am=1,
        stride_ak=K,
        stride_wk=1,
        stride_wn=N,
        stride_outm=1,
        stride_outn=N,
        eps=1e-5,
    )

    return out


# ------------------ New model using the Triton kernel ------------------
class ModelNew(nn.Module):
    """
    Same functional behavior as the original model but with a fused
    Triton kernel for the GEMM + scaling step.
    """

    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)  # bias handled by BN
        self.scale = nn.Parameter(torch.randn(scale_shape, device=self.gemm.weight.device))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous for Triton
        x = x.contiguous()
        # GEMM + scaling fused kernel
        out = triton_gemm_scaled(x, self.gemm.weight, self.scale)
        # BatchNorm
        out = self.bn(out)
        return out