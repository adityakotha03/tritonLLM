import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn.parameter import Parameter


# --------------------------------------------------------------------
# Triton kernel: matmul + add + Swish + Tanh + GELU + Hardtanh
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_fused_kernel(
    A_ptr,
    B_ptr,
    add_ptr,          # (N,) vector
    out_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Each program processes a tile of size BLOCK_SIZE_M x BLOCK_SIZE_N.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the tile indices
    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N

    # Allocate accumulator in registers (float32 for accumulation)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_off = k

        # Load A tile [M, K]
        A_tile = tl.load(
            A_ptr + row * K + k_off + tl.arange(0, BLOCK_SIZE_M)[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :],
            mask=(
                (row + tl.arange(0, BLOCK_SIZE_M)[:, None] < M)
                & (k_off + tl.arange(0, BLOCK_SIZE_K)[None, :] < K)
            ),
            other=0.0,
        ).to(tl.float32)

        # Load B tile [K, N]
        B_tile = tl.load(
            B_ptr + k_off * N + col + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :],
            mask=(
                (k_off + tl.arange(0, BLOCK_SIZE_K)[:, None] < K)
                & (col + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
            ),
            other=0.0,
        ).to(tl.float32)

        # Dot product using tensor cores if possible
        acc += tl.dot(A_tile, B_tile)

    # Add bias vector (add_ptr) - broadcast over rows
    add_vec = tl.load(add_ptr + col + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=(col + tl.arange(0, BLOCK_SIZE_N)[None, :] < N), other=0.0)

    acc = acc + add_vec

    # Swish: x * sigmoid(x)
    acc = acc * tl.math.sigmoid(acc)

    # Tanh
    acc = tl.math.tanh(acc)

    # GELU (approximation)
    acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * tl.math.pow(acc, 3.0))))

    # Hardtanh: clamp between -1 and 1
    acc = tl.math.max(tl.math.min(acc, 1.0), -1.0)

    # Store the result
    tl.store(
        out_ptr + row * N + col + tl.arange(0, BLOCK_SIZE_M)[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :],
        acc,
        mask=(
            (row + tl.arange(0, BLOCK_SIZE_M)[:, None] < M)
            & (col + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
        ),
    )


# --------------------------------------------------------------------
# Triton wrapper for the fused kernel
# --------------------------------------------------------------------
def triton_matmul_fused(A: torch.Tensor, B: torch.Tensor, add_vec: torch.Tensor) -> torch.Tensor:
    """
    A: (M, K)
    B: (K, N)
    add_vec: (N,)
    Returns: (M, N)
    """
    assert A.is_cuda and B.is_cuda and add_vec.is_cuda
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb
    assert add_vec.shape == (N,)

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Compute grid size
    grid = (
        ( (M + 127) // 128, (N + 127) // 128 ),
    )

    matmul_fused_kernel[grid](
        A, B, add_vec, out,
        M, N, K,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )
    return out


# --------------------------------------------------------------------
# New model using the Triton fused kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, add_value_shape):
        super(ModelNew, self).__init__()
        self.weight = Parameter(torch.randn(out_features, in_features, device="cuda", dtype=torch.float16))
        self.bias = Parameter(torch.randn(out_features, device="cuda", dtype=torch.float16))
        # The 'add_value' in the original model is a vector added after matmul.
        # It is the same as bias in this context.
        self.add_value = Parameter(torch.randn(add_value_shape, device="cuda", dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)
        """
        # Convert inputs to fp16 for tensor core
        x = x.to(torch.float16)
        weight = self.weight.to(torch.float16)
        add_vec = self.add_value.to(torch.float16)

        # Matmul + fused ops
        out = triton_matmul_fused(x, weight.t(), add_vec)

        # Convert back to float32 if desired
        return out.to(torch.float32)