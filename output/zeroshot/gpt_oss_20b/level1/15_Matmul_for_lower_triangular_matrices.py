import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel that multiplies two lower‑triangular matrices and writes
# only the lower triangular part of the result.
@triton.jit
def tril_matmul_kernel(
    A_ptr,          # Pointer to first input matrix A
    B_ptr,          # Pointer to second input matrix B
    C_ptr,          # Pointer to output matrix C
    N: tl.constexpr,    # Matrix dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for i and j dimensions
    KB: tl.constexpr,          # Tile size for the k dimension
):
    # Compute the i and j coordinates of this program
    i_start = tl.program_id(0) * BLOCK_SIZE
    j_start = tl.program_id(1) * BLOCK_SIZE

    # Offsets for all elements in the block
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)

    # Mask for elements that are in bounds and in the lower triangle
    mask_ij = (i_offsets[:, None] < N) & (j_offsets[None, :] < N) & (j_offsets[None, :] <= i_offsets[:, None])

    # Accumulator for the block
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    # k iterates over the range j <= k <= i (for each block)
    k_start = j_start
    k_end   = i_start + BLOCK_SIZE
    k_end   = min(k_end, N)

    for k in range(k_start, k_end, KB):
        k_offsets = k + tl.arange(0, KB)

        # Masks for A[i,k] and B[k,j] to respect the triangular structure
        a_mask = (i_offsets[:, None] >= k_offsets[None, :]) & (k_offsets[None, :] < N)
        b_mask = (k_offsets[:, None] >= j_offsets[None, :]) & (k_offsets[None, :] < N)

        # Load tiles with proper masking
        a = tl.load(A_ptr + i_offsets[:, None] * N + k_offsets[None, :], mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offsets[:, None] * N + j_offsets[None, :], mask=b_mask, other=0.0)

        # Accumulate
        acc += a * b

    # Store the result (only the lower triangular part)
    tl.store(C_ptr + i_offsets[:, None] * N + j_offsets[None, :], acc, mask=mask_ij)


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Multiply two lower‑triangular matrices A and B and return the lower‑triangular
    part of the product.  A and B must be square tensors of the same shape.
    """
    assert A.shape == B.shape, "Input matrices must have the same shape."
    N = A.shape[0]
    # Ensure contiguous memory
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    C = torch.empty_like(A)

    # Kernel parameters
    BLOCK_SIZE = 64   # Tune this value for best performance
    KB         = 64   # Tile size for the k dimension

    # Grid definition
    grid = lambda meta: (
        (N + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )

    # Launch Triton kernel
    tril_matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE, KB=KB)

    return C


class ModelNew(nn.Module):
    """
    Optimized model that multiplies two lower‑triangular matrices using a
    custom Triton kernel.  The result is automatically the lower‑triangular
    part of the product.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_tril_matmul(A, B)