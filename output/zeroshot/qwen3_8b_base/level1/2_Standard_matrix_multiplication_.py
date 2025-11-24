import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,  # Pointers to A, B, and C
    M, N, K,  # Dimensions of the matrices
    stride_am, stride_ak,  # Strides of A
    stride_bk, stride_bn,  # Strides of B
    stride_cm, stride_cn,  # Strides of C
    BLOCK_SIZE: tl.constexpr,
):
    # Use the block index to compute the block position in the matrix
    pid = tl.program_id(0)
    # Compute the block's (m, n) position in the matrix
    block_m = pid // (tl.num_programs(0) // tl.num_programs(1))
    block_n = pid % (tl.num_programs(0) // tl.num_programs(1))
    # Compute the block's (m, n) offset
    m = block_m * BLOCK_SIZE
    n = block_n * BLOCK_SIZE
    # Compute the block's (k) range
    k_start = 0
    k_end = K
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Iterate over the k dimension
    for k in range(k_start, k_end, BLOCK_SIZE):
        # Load A and B blocks
        a = tl.load(A + m + k * stride_am, None, (BLOCK_SIZE, BLOCK_SIZE))
        b = tl.load(B + k + n * stride_bk, None, (BLOCK_SIZE, BLOCK_SIZE))
        # Compute the dot product
        acc += tl.dot(a, b)
    # Store the result
    tl.store(C + m + n * stride_cm, acc, None)


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
    C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

    # Dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions do not match."

    # Grid size
    num_blocks_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks_m * num_blocks_n, )

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, M, N, K, stride_am=A.stride(0), stride_ak=A.stride(1),
                        stride_bk=B.stride(0), stride_bn=B.stride(1),
                        stride_cm=C.stride(0), stride_cn=C.stride(1),
                        BLOCK_SIZE=1024)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)