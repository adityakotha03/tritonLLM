import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A,  # (M, K)
    B,  # (K, N)
    C,  # (M, N)
    M,  # M dimension
    K,  # K dimension
    N,  # N dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of M x N
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE)

    # Compute the block indices
    m = pid // (num_pid_n * num_pid_k)
    n = (pid // num_pid_k) % num_pid_n
    k = pid % num_pid_k

    # Compute the block offsets
    offs_m = m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load A and B
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :], mask=offs_k < K, other=0.0)
    b = tl.load(B + offs_k[None, :] * N + offs_n[:, None], mask=offs_n < N, other=0.0)

    # Compute the dot product
    c = tl.dot(a, b)

    # Store the result
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c, mask=offs_m < M and offs_n < N)


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

    # Determine the block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_pid_m = tl.cdiv(A.shape[0], BLOCK_SIZE)
    num_pid_n = tl.cdiv(B.shape[1], BLOCK_SIZE)
    num_pid_k = tl.cdiv(A.shape[1], BLOCK_SIZE)

    # Launch the Triton kernel
    grid = (num_pid_m * num_pid_n * num_pid_k,)
    matmul_kernel[grid](A, B, C, A.shape[0], A.shape[1], B.shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Instead of using torch.matmul, call our Triton-based matrix multiplication
        return triton_matmul(A, B)