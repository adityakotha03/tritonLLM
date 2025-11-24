import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a tile of data
    pid = tl.program_id(0)
    # Compute the block (i, j) in the output matrix
    i = pid // M
    j = pid % M
    # Compute the offset in the output matrix
    offset = i * M * N + j * N
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # Iterate over the k dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load A and B blocks
        a = tl.load(A_ptr + i * K + k + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < K - k, other=0.0)
        b = tl.load(B_ptr + k + j * K + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < K - k, other=0.0)
        # Compute the dot product
        acc += tl.dot(a, b)
    # Store the result
    tl.store(C_ptr + offset + tl.arange(0, BLOCK_SIZE), acc, mask=tl.arange(0, BLOCK_SIZE) < N)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom matrix multiplication using Triton.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[0] == B.shape[1], "Matrix dimensions must be compatible."
    assert A.shape[1] == B.shape[0], "Matrix dimensions must be compatible."
    N = A.shape[0]
    K = A.shape[1]
    M = B.shape[1]
    # Output tensor
    C = torch.empty((N, M), dtype=A.dtype, device=A.device)
    # Determine block size
    BLOCK_SIZE = 128
    # Compute number of blocks
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    grid = (num_blocks,)
    matmul_kernel[grid](A, B, C, N, K, M, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)