import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_scalar_kernel(
    A_ptr,  # Pointer to input matrix A
    out_ptr,  # Pointer to output matrix
    s,  # Scalar value
    M,  # Number of rows in A
    N,  # Number of columns in A
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns
):
    # Calculate the program ID and the starting row index for this block
    pid_m = tl.program_id(0)  # Program ID along M dimension
    pid_n = tl.program_id(1)  # Program ID along N dimension

    # Calculate the starting offsets for this block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create index arrays for rows and columns within this block
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create masks to handle boundary conditions (if block extends beyond matrix bounds)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load the input matrix block
    A = tl.load(A_ptr + offsets_m[:, None] * N + offsets_n[None, :], mask=mask, other=0.0)

    # Perform elementwise multiplication by the scalar
    out = A * s

    # Store the result
    tl.store(out_ptr + offsets_m[:, None] * N + offsets_n[None, :], out, mask=mask)


def triton_matmul_scalar(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Custom Triton kernel wrapper for matrix-scalar multiplication.

    Args:
        A: Input tensor of shape (M, N)
        s: Scalar value

    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda, "Input tensor must be on CUDA device"
    A = A.contiguous()

    M, N = A.shape
    out = torch.empty_like(A)

    # Define block sizes optimized for A100
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256

    # Calculate the number of program instances (blocks)
    num_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)

    # Define the grid configuration
    grid = lambda meta: (num_blocks_m, num_blocks_n)

    # Launch the Triton kernel
    matmul_scalar_kernel[grid](
        A, out, s, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using a custom Triton kernel.
        """
        return triton_matmul_scalar(A, s)