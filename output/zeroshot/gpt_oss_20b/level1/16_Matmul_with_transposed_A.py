import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel for matrix multiplication C = A.T @ B
# A: (K, M)  (not transposed, we read A[k, i] as A[k, i])
# B: (K, N)
# Output C: (M, N)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Starting indices for this program
    row_base = tl.program_id(0) * BLOCK_M
    col_base = tl.program_id(1) * BLOCK_N

    # Accumulator for this tile (float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k dimension
    for k in range(0, K, BLOCK_K):
        # Current block of k indices
        k_start = k
        k_end = min(k_start + BLOCK_K, K)
        k_offset = tl.arange(0, BLOCK_K)[:, None]  # shape (BLOCK_K, 1)

        # ----- Load tile of A: shape (BLOCK_K, BLOCK_M) ----------
        # Compute offsets for A[k,i] = k * stride_am + i
        row_offset = row_base + tl.arange(0, BLOCK_M)[None, :]  # shape (1, BLOCK_M)
        A_offsets = (k_start + k_offset) * stride_am + row_offset  # broadcasted
        # Masks for boundary conditions
        mask_k_a = (k_start + k_offset) < K
        mask_m_a = row_base + tl.arange(0, BLOCK_M) < M
        mask_a = mask_k_a & mask_m_a[:, None]
        # Load A tile
        A_tile = tl.load(
            A_ptr + A_offsets, mask=mask_a, other=0.0, dtype=tl.float16
        )

        # ----- Load tile of B: shape (BLOCK_K, BLOCK_N) ----------
        col_offset = col_base + tl.arange(0, BLOCK_N)[None, :]  # shape (1, BLOCK_N)
        B_offsets = (k_start + k_offset) * stride_bn + col_offset  # broadcasted
        # Masks for boundary conditions
        mask_k_b = (k_start + k_offset) < K
        mask_n_b = col_base + tl.arange(0, BLOCK_N) < N
        mask_b = mask_k_b & mask_n_b[:, None]
        # Load B tile
        B_tile = tl.load(
            B_ptr + B_offsets, mask=mask_b, other=0.0, dtype=tl.float16
        )

        # ----- Compute partial product and accumulate ------------
        # A_tile: (BLOCK_K, BLOCK_M), B_tile: (BLOCK_K, BLOCK_N)
        # We need to compute A_tile.T @ B_tile -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(A_tile.T, B_tile, dtype=tl.float32)

    # ----- Store the result ------------------------------------
    # Compute final offsets for C[i,j] = i * N + j
    i_offset = row_base + tl.arange(0, BLOCK_M)[None, :]  # shape (1, BLOCK_M)
    j_offset = col_base + tl.arange(0, BLOCK_N)[None, :]  # shape (1, BLOCK_N)
    C_offsets = i_offset * N + j_offset
    # Mask for output boundaries
    mask_m = row_base + tl.arange(0, BLOCK_M) < M
    mask_n = col_base + tl.arange(0, BLOCK_N) < N
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + C_offsets, acc, mask=mask_c)


# ----------------------------------------------------------------------
# Helper function to launch the Triton matmul kernel
# ----------------------------------------------------------------------
def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform C = A.T @ B using a Triton kernel.
    A: (K, M) float32 tensor
    B: (K, N) float32 tensor
    Returns:
        C: (M, N) float32 tensor
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    K, M = A.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimension must match."

    # Convert to fp16 for computation (no extra copy if already fp16)
    A_fp16 = A.to(torch.float16).contiguous()
    B_fp16 = B.to(torch.float16).contiguous()

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Strides for accessing A[k, i] and B[k, j]
    stride_am = M  # each row of A has M columns
    stride_bn = N  # each row of B has N columns

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        A_fp16,
        B_fp16,
        C,
        M,
        N,
        K,
        stride_am,
        stride_bn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


# ----------------------------------------------------------------------
# Model definition using the custom Triton kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication
    C = A.T @ B, where A: (K, M), B: (K, N)
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# ----------------------------------------------------------------------
# Example usage (uncomment for testing, but not part of the submission)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     M = 1024 * 2
#     K = 4096 * 2
#     N = 2048 * 2
#     A = torch.rand(K, M, device="cuda")
#     B = torch.rand(K, N, device="cuda")
#     model = ModelNew().cuda()
#     out = model(A, B)
#     print(out.shape)