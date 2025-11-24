import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to first input (N, M, K)
    B_ptr,  # Pointer to second input (K, L)
    C_ptr,  # Pointer to output (N, M, L)
    N, M, K, L,  # Dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the block index in the output (N, M, L)
    n_idx = pid // (M * L)
    m_idx = (pid // L) % M
    l_idx = pid % L

    # Compute the offset in the output
    offset = n_idx * M * L + m_idx * L + l_idx

    # Compute the start index in A and B
    a_start = n_idx * M * K + m_idx * K
    b_start = l_idx * K

    # Compute the range of offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    a_offsets = a_start + offsets
    b_offsets = b_start + offsets

    # Load A and B
    a = tl.load(A_ptr + a_offsets, mask=offsets < K, other=0.0)
    b = tl.load(B_ptr + b_offsets, mask=offsets < K, other=0.0)

    # Compute the dot product
    c = tl.dot(a, b)

    # Store the result
    tl.store(C_ptr + offset, c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    C = torch.empty((A.size(0), A.size(1), B.size(1)), device=A.device, dtype=A.dtype)

    # Dimensions
    N, M, K = A.shape
    _, L = B.shape

    # Choose block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (N * M * L,)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, N, M, K, L, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)