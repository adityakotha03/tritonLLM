import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_tril_kernel(
    a_ptr, b_ptr, c_ptr,
    N: tl.constexpr,
    stride_a_row, stride_b_row, stride_c_row,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Program ID along row dimension
    pid = tl.program_id(0)
    # Block offset
    block_base = pid * BLOCK_SIZE
    # Offset for the block
    row_offsets = block_base + tl.arange(0, BLOCK_SIZE)
    # Mask for valid row indices
    row_mask = row_offsets < N

    # Load A block (A is lower triangular)
    a_block = tl.load(
        a_ptr + row_offsets[:, None] * stride_a_row + tl.arange(0, N)[None, :],
        mask=row_mask[:, None] & (tl.arange(0, N)[None, :] <= row_offsets[:, None]),
        other=0.0
    )

    # Initialize accumulator for result
    acc = tl.zeros((BLOCK_SIZE, TILE_SIZE), dtype=tl.float32)

    # Iterate over tiles of B
    for start in range(0, N, TILE_SIZE):
        # Compute column offsets for current tile
        col_offsets = start + tl.arange(0, TILE_SIZE)
        col_mask = col_offsets < N

        # Load B block (B is lower triangular)
        b_block = tl.load(
            b_ptr + tl.arange(0, N)[:, None] * stride_b_row + col_offsets[None, :],
            mask=(tl.arange(0, N)[:, None] <= tl.arange(0, N)[:, None]) & (col_offsets[None, :] < N),
            other=0.0
        )

        # Perform block-wise matrix multiplication
        acc += tl.dot(a_block, b_block)

    # Store result, but ensure lower triangular structure
    # Only store where row >= col
    col_indices = tl.arange(0, BLOCK_SIZE)[:, None]  # Broadcast row indices
    row_indices = tl.arange(0, TILE_SIZE)[None, :]   # Broadcast col indices
    mask = (row_indices < N) & (col_indices < N) & (col_indices <= row_indices)  # Only lower triangular
    tl.store(
        c_ptr + row_offsets[:, None] * stride_c_row + col_indices[None, :],
        acc,
        mask=mask
    )


def triton_matmul_tril(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape == B.shape, "A and B must have the same shape"
    N = A.shape[0]
    assert N % 128 == 0, "N must be divisible by 128 for optimal block size"

    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    C = torch.empty_like(A)

    # Block size and tile size
    BLOCK_SIZE = 128
    TILE_SIZE = 128

    # Grid definition
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # Launch kernel
    matmul_tril_kernel[grid](
        A, B, C,
        N=N,
        stride_a_row=A.stride(0),
        stride_b_row=B.stride(0),
        stride_c_row=C.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul_tril(A, B)