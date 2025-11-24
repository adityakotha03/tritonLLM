import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to input A (M, K)
    b_ptr,  # Pointer to input B (K, N)
    c_ptr,  # Pointer to output C (M, N)
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program instance handles a block of rows from A and columns from B
    pid_m = tl.program_id(0)  # Row index of the block in A
    pid_n = tl.program_id(1)  # Column index of the block in B

    # Compute the starting row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Create a range of offsets for the current block
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)

    # Load A and B in tiles
    # A: (M, K) -> load rows in blocks of BLOCK_SIZE_M
    # B: (K, N) -> load columns in blocks of BLOCK_SIZE_N
    a = tl.load(a_ptr + offsets_m[:, None] * k + offsets_m[:, None], mask=offsets_m[:, None] < m, other=0.0)
    b = tl.load(b_ptr + offsets_n[None, :] * n + offsets_n[None, :], mask=offsets_n[None, :] < n, other=0.0)

    # Compute the dot product across K
    # We use a temporary accumulator to compute the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k_idx in range(0, k, BLOCK_SIZE_M):
        # Load a slice of A and B
        a_slice = tl.load(a_ptr + (offsets_m[:, None] * k + k_idx), mask=offsets_m[:, None] < m, other=0.0)
        b_slice = tl.load(b_ptr + (k_idx + offsets_n[None, :] * n), mask=offsets_n[None, :] < n, other=0.0)
        acc = acc + tl.dot(a_slice, b_slice)

    # Store the result in C
    c = tl.load(c_ptr + offsets_m[:, None] * n + offsets_n[None, :], mask=offsets_m[:, None] < m, other=0.0)
    c = c + acc
    tl.store(c_ptr + offsets_m[:, None] * n + offsets_n[None, :], c, mask=offsets_m[:, None] < m)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication with optimized tiling and tensor core usage.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    m, k = a.shape
    k, n = b.shape

    # Output tensor
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)

    # Define block sizes (powers of 2 for optimal performance)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Grid dimensions: number of blocks in M and N directions
    grid = lambda meta: (
        (m + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (n + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    # Launch the kernel with autotuning on block sizes
    matmul_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=c.data_ptr(),
        m=m,
        k=k,
        n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)