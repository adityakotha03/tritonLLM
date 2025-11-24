import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
#   Triton kernel that fuses matrix multiplication, ReLU and division
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128},
                      num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_matmul_relu_div_kernel(
    A_ptr,          # [M, K]
    B_ptr,          # [K, N]
    out_ptr,        # [M, N]
    M, N, K,        # dimensions
    divisor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column indices for this block
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Allocate registers for the partial results
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # Load tile of A (M x K) and B (K x N)
        A_block = tl.load(
            A_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * K
            + (k + tl.arange(0, BLOCK_K))[None, :],
            mask=(row_start + tl.arange(0, BLOCK_M)[:, None] < M)
                 & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        B_block = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N
            + (col_start + tl.arange(0, BLOCK_N))[None, :],
            mask=(k + tl.arange(0, BLOCK_K)[:, None] < K)
                 & (col_start + tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(A_block, B_block)

    # Apply ReLU
    acc = tl.max(acc, 0.0)

    # Scale by divisor
    acc = acc / divisor

    # Store result
    tl.store(
        out_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * N
        + (col_start + tl.arange(0, BLOCK_N))[None, :],
        acc,
        mask=(row_start + tl.arange(0, BLOCK_M)[:, None] < M)
             & (col_start + tl.arange(0, BLOCK_N)[None, :] < N),
    )

# --------------------------------------------------------------------------- #
#   Helper that runs the kernel for a batch
# --------------------------------------------------------------------------- #
def fused_matmul_relu_div(A: torch.Tensor, B: torch.Tensor, divisor: float):
    """
    A: [M, K] (float16 or bfloat16)
    B: [K, N] (float16 or bfloat16)
    Returns: [M, N] (float16)
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    fused_matmul_relu_div_kernel[grid](
        A, B, out,
        M, N, K,
        divisor=divisor,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return out

# --------------------------------------------------------------------------- #
#   Model that uses the Triton fused kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised model that fuses linear, ReLU and division into a single Triton kernel.
    """
    def __init__(self, in_features: int, out_features: int, divisor: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor

        # Weights and bias are stored in bfloat16 for reduced memory traffic
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        )
        self.bias = nn.Parameter(
            torch.randn(out_features, device="cuda", dtype=torch.bfloat16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, in_features]  (float32 or bfloat16)
        Returns: [batch, out_features] (float16)
        """
        assert x.is_cuda

        # Cast input to bfloat16 if needed
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Matrix multiplication
        out = fused_matmul_relu_div(x, self.weight.t(), self.divisor)

        # Add bias after ReLU/division (since ReLU div is element‑wise)
        out = out + self.bias.unsqueeze(0)

        return out