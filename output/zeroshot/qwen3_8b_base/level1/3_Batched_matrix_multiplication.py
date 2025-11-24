import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,  # Pointer to A tensor (batch_size, m, k)
    B_ptr,  # Pointer to B tensor (batch_size, k, n)
    C_ptr,  # Pointer to C tensor (batch_size, m, n)
    batch_size,  # Number of batches
    m,  # m dimension
    n,  # n dimension
    k,  # k dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program processes a block of C
    pid = tl.program_id(0)
    # Compute the batch index
    batch_idx = pid // (BLOCK_M * BLOCK_N)
    # Compute the block index within the batch
    block_idx = pid % (BLOCK_M * BLOCK_N)
    # Compute the row and column indices within the block
    row_idx = block_idx // BLOCK_N
    col_idx = block_idx % BLOCK_N

    # Compute the offset for A and B
    a_offsets = tl.arange(0, BLOCK_K)
    b_offsets = tl.arange(0, BLOCK_K)
    c_offsets = row_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # Compute the offset for the batch
    batch_offset = batch_idx * m * k * n

    # Compute the offset for A
    a_offsets = batch_offset + row_idx * k * n + a_offsets * n + col_idx * n
    # Compute the offset for B
    b_offsets = batch_offset + col_idx * k * n + b_offsets * n + row_idx * n
    # Compute the offset for C
    c_offsets = batch_offset + row_idx * n + c_offsets

    # Load A and B
    a = tl.load(A_ptr + a_offsets, mask=a_offsets < (batch_idx + 1) * m * k * n, other=0.0)
    b = tl.load(B_ptr + b_offsets, mask=b_offsets < (batch_idx + 1) * m * k * n, other=0.0)

    # Compute the dot product
    c = tl.dot(a, b)

    # Store the result
    tl.store(C_ptr + c_offsets, c, mask=c_offsets < (batch_idx + 1) * m * n)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor):
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
    C = torch.empty((A.size(0), A.size(1), B.size(2)), dtype=A.dtype, device=A.device)

    # Number of elements in the tensor
    batch_size = A.size(0)
    m = A.size(1)
    n = B.size(2)
    k = A.size(2)

    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128

    # Determine the number of blocks needed
    num_blocks = (batch_size * m * n + BLOCK_M * BLOCK_N - 1) // (BLOCK_M * BLOCK_N)

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    batched_matmul_kernel[grid](A, B, C, batch_size, m, n, k, BLOCK_M, BLOCK_N, BLOCK_K)

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_batched_matmul(A, B)