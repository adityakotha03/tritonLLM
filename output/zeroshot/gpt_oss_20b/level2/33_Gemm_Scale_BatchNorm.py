import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernel for matrix multiplication with optional scaling
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 8}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_fused_kernel(
    A_ptr,  # shape [M, K]
    B_ptr,  # shape [K, N]
    out_ptr,  # shape [M, N]
    scale_ptr,  # shape [N] (optional, None -> no scaling)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE: tl.constexpr,  # 1 if scaling enabled, 0 otherwise
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the starting row/col for this block
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Allocate registers for the output tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B
        A = tl.load(
            A_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * K
            + (k + tl.arange(0, BLOCK_K))[None, :],
            mask=(row_start + tl.arange(0, BLOCK_M))[:, None] < M
            & (k + tl.arange(0, BLOCK_K))[None, :] < K,
            other=0.0,
        )
        B = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N
            + (col_start + tl.arange(0, BLOCK_N))[None, :],
            mask=(k + tl.arange(0, BLOCK_K))[:, None] < K
            & (col_start + tl.arange(0, BLOCK_N))[None, :] < N,
            other=0.0,
        )
        acc += tl.dot(A, B)

    # Scale by per-column scale factor if enabled
    if SCALE:
        scale = tl.load(scale_ptr + col_start + tl.arange(0, BLOCK_N))
        acc *= scale[None, :]

    # Store the result
    tl.store(
        out_ptr
        + (row_start + tl.arange(0, BLOCK_M))[:, None] * N
        + (col_start + tl.arange(0, BLOCK_N))[None, :],
        acc,
        mask=(row_start + tl.arange(0, BLOCK_M))[:, None] < M
        & (col_start + tl.arange(0, BLOCK_N))[None, :] < N,
    )


def matmul_fused(A: torch.Tensor, B: torch.Tensor, scale: torch.Tensor | None = None):
    """
    Performs A @ B (both in bf16) optionally multiplied by a column‑wise scale.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."
    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)
    out = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Grid definition: two‑dimensional grid over output tiles
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    _matmul_fused_kernel[grid](
        A, B, out,
        scale if scale is not None else torch.empty(0, device=A.device),
        M, N, K,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
        SCALE=1 if scale is not None else 0,
    )
    return out


# --------------------------------------------------------------------------- #
# ModelNew with Triton GEMM + scaling
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised model that replaces the linear layer with a Triton GEMM,
    then applies a per‑output scaling and batch‑norm.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        # Weight and bias for linear layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(out_features, device="cuda"))
        # Scaling factor (per output feature)
        self.scale = nn.Parameter(torch.randn(scale_shape, device="cuda"))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor):
        # x: [B, in_features]
        # Perform GEMM with Triton
        mat = matmul_fused(x, self.weight.t(), self.scale)
        # Add bias
        mat = mat + self.bias
        # BatchNorm
        out = self.bn(mat)
        return out