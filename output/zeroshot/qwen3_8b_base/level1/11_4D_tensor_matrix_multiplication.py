import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to 4D tensor A (b, i, j, l)
    B_ptr,  # Pointer to matrix B (l, k)
    C_ptr,  # Pointer to output tensor C (b, i, j, k)
    b, i, j, k, l,  # Dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block index in the output tensor
    block_b = pid // (i * j * k)
    block_i = (pid // (j * k)) % i
    block_j = (pid // k) % j
    block_k = pid % k

    # Compute the offset in the output tensor
    offset = block_b * i * j * k + block_i * j * k + block_j * k + block_k
    # Compute the starting index in A and B
    a_start = block_b * i * j * l + block_i * j * l + block_j * l
    b_start = block_k * l

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the l dimension
    for l_idx in range(0, l, BLOCK_SIZE):
        # Load A and B blocks
        a_block = tl.load(A_ptr + a_start + l_idx, mask=(l_idx + BLOCK_SIZE <= l), other=0.0)
        b_block = tl.load(B_ptr + b_start + l_idx, mask=(l_idx + BLOCK_SIZE <= l), other=0.0)
        # Compute the dot product
        acc += tl.dot(a_block, b_block)
    
    # Store the result
    tl.store(C_ptr + offset, acc)


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
    C = torch.empty((A.size(0), A.size(1), A.size(2), B.size(1)), dtype=A.dtype, device=A.device)

    # Dimensions
    b, i, j, l = A.shape
    k = B.shape[1]

    # Choose optimal block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks = (i * j * k + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    matmul_kernel[ num_blocks ](A, B, C, b, i, j, k, l, l, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Instead of "return torch.einsum("bijl,lk->bijk", A, B)", call our Triton-based matmul
        return triton_matmul(A, B)