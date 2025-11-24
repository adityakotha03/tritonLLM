import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to vector B
    C_ptr,  # Pointer to output vector C
    M: tl.constexpr,  # Number of rows in A
    K: tl.constexpr,  # Number of columns in A (rows in B)
    N: tl.constexpr,  # Number of columns in B (1 in this case)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of M elements
    pid = tl.program_id(0)
    # Compute the row index of the current program
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = row < M

    # Load vector B into a register
    b = tl.load(B_ptr, mask=mask, other=0.0)

    # Compute the dot product for each row
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(A_ptr + k + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        acc += a * b

    # Store the result
    tl.store(C_ptr + row, acc, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton kernel for matrix-vector multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor of shape (M, 1)
    C = torch.empty((A.size(0), 1), device=A.device, dtype=A.dtype)

    # Define block size (must be power of 2)
    BLOCK_SIZE = 128

    # Grid size (number of blocks)
    grid = lambda meta: ((A.size(0) + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    matmul_kernel[grid](A, B, C, M=A.size(0), K=A.size(1), N=1, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)