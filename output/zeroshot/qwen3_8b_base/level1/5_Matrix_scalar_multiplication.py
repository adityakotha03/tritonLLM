import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mult_kernel(
    A_ptr,  # Pointer to input matrix A
    s_ptr,  # Pointer to scalar s
    out_ptr,  # Pointer to output matrix
    M,  # Number of rows
    N,  # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the row index for this program
    row = pid * BLOCK_SIZE
    # Create a range of column indices [0..BLOCK_SIZE-1]
    cols = tl.arange(0, BLOCK_SIZE)
    # Compute the offset for the current row and column
    offsets = row * N + cols
    # Mask to ensure we don't go out of bounds
    mask = (row < M) & (cols < N)
    # Load scalar value
    s = tl.load(s_ptr)
    # Load input values
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise multiplication
    out = A * s
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scalar_mult(A: torch.Tensor, s: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()
    # Prepare output tensor
    out = torch.empty_like(A)
    # Number of rows and columns
    M = A.shape[0]
    N = A.shape[1]
    # Choose a block size that's a power of two and fits within the GPU's capabilities
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    scalar_mult_kernel[grid](A, torch.tensor(s).cuda(), out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # Instead of "return A * s", call our Triton-based scalar multiplication
        return triton_scalar_mult(A, s)