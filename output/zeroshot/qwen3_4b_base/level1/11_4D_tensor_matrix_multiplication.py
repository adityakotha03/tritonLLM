import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to input tensor A of shape (b, i, j, l)
    B_ptr,  # Pointer to input matrix B of shape (l, k)
    C_ptr,  # Pointer to output tensor C of shape (b, i, j, k)
    b, i, j, l, k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one block of (i, j) indices
    b_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)

    # Compute the global indices
    b_start = b_idx * (b // BLOCK_SIZE) if b_idx < b else 0
    i_start = i_idx * (i // BLOCK_SIZE) if i_idx < i else 0
    j_start = j_idx * (j // BLOCK_SIZE) if j_idx < j else 0

    # Define the block size for each dimension
    i_block_size = min(BLOCK_SIZE, i - i_start)
    j_block_size = min(BLOCK_SIZE, j - j_start)

    # Create offsets for the current block
    i_offsets = tl.arange(0, i_block_size)
    j_offsets = tl.arange(0, j_block_size)
    l_offsets = tl.arange(0, l)

    # Load A: (b, i, j, l) -> use tiling to load in blocks
    # A[b, i, j, l] -> A_ptr + b_start + i_start + j_start + l
    A = tl.zeros((i_block_size, j_block_size, l), dtype=tl.float16)
    B = tl.zeros((l, k), dtype=tl.float16)

    # Load A in tile
    for ii in range(i_block_size):
        for jj in range(j_block_size):
            i_pos = i_start + ii
            j_pos = j_start + jj
            # Load A[b, i_pos, j_pos, :] for all l
            A_idx = tl.arange(0, l)
            A_vals = tl.load(A_ptr + b * i * j + b_idx * i * j + i_pos * j + j_pos * l + A_idx, mask=A_idx < l, other=0.0)
            A[ii, jj, :] = A_vals

    # Load B in tile
    B_offsets = tl.arange(0, l)
    B_vals = tl.load(B_ptr + B_offsets, mask=B_offsets < l, other=0.0)
    B = B_vals

    # Compute output C[b, i, j, k]
    C = tl.zeros((i_block_size, j_block_size, k), dtype=tl.float16)
    for kk in range(k):
        # Compute dot product over l
        for ll in range(l):
            # Load B[ll, kk]
            b_val = B[ll, kk]
            # Load A[i, j, ll]
            # We have A[i, j, ll] already loaded in A[ii, jj, ll]
            A_val = A[i_offsets, j_offsets, ll]
            C[i_offsets, j_offsets, kk] += A_val * b_val

    # Store result
    C_ptr_base = C_ptr + b_idx * i * j * k + i_start * j * k + j_start * k
    C_offsets = tl.arange(0, i_block_size) * j_block_size * k + tl.arange(0, j_block_size) * k + tl.arange(0, k)
    C_mask = C_offsets < i_block_size * j_block_size * k
    tl.store(C_ptr_base + C_offsets, C, mask=C_mask)


@triton.jit
def matmul_kernel_fused(
    A_ptr,
    B_ptr,
    C_ptr,
    b, i, j, l, k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of (i, j) indices
    b_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)

    # Compute block indices
    i_start = i_idx * BLOCK_SIZE
    j_start = j_idx * BLOCK_SIZE
    b_start = b_idx * (b // BLOCK_SIZE)

    # Define block size
    i_block_size = min(BLOCK_SIZE, i - i_start)
    j_block_size = min(BLOCK_SIZE, j - j_start)

    # Create offsets
    i_offsets = tl.arange(0, i_block_size)
    j_offsets = tl.arange(0, j_block_size)
    l_offsets = tl.arange(0, l)

    # Load A: (b, i, j, l)
    A = tl.zeros((i_block_size, j_block_size, l), dtype=tl.float16)
    for ii in range(i_block_size):
        for jj in range(j_block_size):
            i_pos = i_start + ii
            j_pos = j_start + jj
            # Load A[b, i_pos, j_pos, :]
            A_idx = tl.arange(0, l)
            A_vals = tl.load(A_ptr + b_start * i * j * l + i_pos * j * l + j_pos * l + A_idx, mask=A_idx < l, other=0.0)
            A[ii, jj, :] = A_vals

    # Load B: (l, k)
    B = tl.zeros((l, k), dtype=tl.float16)
    B_offsets = tl.arange(0, l)
    B_vals = tl.load(B_ptr + B_offsets, mask=B_offsets < l, other=0.0)
    B = B_vals

    # Compute output C[b, i, j, k]
    C = tl.zeros((i_block_size, j_block_size, k), dtype=tl.float16)
    for kk in range(k):
        for ll in range(l):
            b_val = B[ll, kk]
            for ii in range(i_block_size):
                for jj in range(j_block_size):
                    A_val = A[ii, jj, ll]
                    C[ii, jj, kk] += A_val * b_val

    # Store result
    C_ptr_base = C_ptr + b_start * i * j * k + i_start * j * k + j_start * k
    C_offsets = i_offsets * j_block_size * k + j_offsets * k + tl.arange(0, k)
    C_mask = C_offsets < i_block_size * j_block_size * k
    tl.store(C_ptr_base + C_offsets, C, mask=C_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication using a custom Triton kernel.
    Optimized for memory coalescing and fused computation.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    b, i, j, l = A.shape
    k = B.shape[1]

    # Define block size (power of 2)
    BLOCK_SIZE = 128

    # Grid dimensions
    grid = lambda meta: (
        (b + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (i + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (j + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    matmul_kernel_fused[grid](A, B, torch.empty_like(A), b, i, j, l, k, BLOCK_SIZE=BLOCK_SIZE)

    return A


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)