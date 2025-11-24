import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_lower_tri_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    N: tl.constexpr,  # Size of square matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_start = pid * BLOCK_SIZE
    # Compute the block's row and column indices
    row_idx = block_start // N
    col_idx = block_start % N
    # Compute the offset for the current block
    offset = row_idx * N + col_idx
    # Create a range of offsets for the current block
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N * N
    # Load A and B
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Compute the dot product
    C = tl.sum(A * B, axis=1)
    # Store the result
    tl.store(C_ptr + offsets, C, mask=mask)


def triton_matmul_lower_tri(A: torch.Tensor, B: torch.Tensor):
    """
    This function wraps the Triton kernel call for matrix multiplication of lower triangular matrices.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    C = torch.empty_like(A)

    # Number of elements in the tensor
    N = A.shape[0]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_lower_tri_kernel[grid](A, B, C, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Replace the original matmul + tril with the Triton kernel
        return triton_matmul_lower_tri(A, B)