import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Program ID along the N dimension
    pid = tl.program_id(0)

    # Offset for the current block
    block_start_n = pid * BLOCK_SIZE_N

    # Create row and column offsets
    row_offsets = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    col_offsets = tl.arange(0, TILE_SIZE)

    # Mask to prevent out-of-bounds access
    row_mask = row_offsets < N
    col_mask = col_offsets < N

    # Accumulate result in register
    acc = tl.zeros((BLOCK_SIZE_N, TILE_SIZE), dtype=tl.float32)

    # Iterate over chunks of B (tile by tile)
    for k in range(0, N, TILE_SIZE):
        # Calculate column index for B
        k_col = k + tl.arange(0, TILE_SIZE)

        # Load A and B in tiles
        A = tl.load(
            A_ptr + (row_offsets[:, None] * N + k_col[None, :]),
            mask=(row_mask[:, None] & col_mask[None, :]),
            other=0.0
        )
        B = tl.load(
            B_ptr + (k_col[:, None] * N + col_offsets[None, :]),
            mask=(col_mask[:, None] & row_mask[None, :]),
            other=0.0
        )

        # Perform matrix multiplication
        acc += tl.dot(A, B)

    # Store the result
    C = acc.to(tl.float32)
    tl.store(
        C_ptr + (row_offsets[:, None] * N + col_offsets[None, :]),
        C,
        mask=(row_mask[:, None] & col_mask[None, :])
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Perform matrix multiplication using Triton kernel.

    Args:
        A (torch.Tensor): Input matrix A, shape (N, N)
        B (torch.Tensor): Input matrix B, shape (N, N)

    Returns:
        torch.Tensor: Output matrix C, shape (N, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape == B.shape, "Input matrices must have the same shape"
    N = A.shape[0]

    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()

    # Initialize output tensor
    C = torch.empty_like(A)

    # Tune block size for optimal performance
    BLOCK_SIZE_N = 256  # Optimized block size for A100
    TILE_SIZE = 256     # Tile size for blocking

    # Grid configuration: number of blocks along the N dimension
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    # Launch the Triton kernel
    matmul_kernel[grid](
        A, B, C,
        N=N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        TILE_SIZE=TILE_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of two symmetric matrices
        using a custom Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul(A, B)