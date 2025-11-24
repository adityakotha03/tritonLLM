import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a 16x16 block of C
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_SIZE)
    num_block_n = tl.cdiv(N, BLOCK_SIZE)
    num_block_k = tl.cdiv(K, BLOCK_SIZE)

    # Compute the block of C
    block_m = pid % num_block_m
    block_n = pid // num_block_m

    # Compute the offset of the block in C
    off_cm = block_m * BLOCK_SIZE
    off_cn = block_n * BLOCK_SIZE

    # Load the block of A
    offs_am = off_cm + tl.arange(0, BLOCK_SIZE)
    offs_ak = tl.arange(0, BLOCK_SIZE)
    a_ptrs = A + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a = tl.load(a_ptrs, mask=offs_ak[None, :] < K, other=0.0)

    # Load the block of B
    offs_bk = tl.arange(0, BLOCK_SIZE)
    offs_bn = off_cn + tl.arange(0, BLOCK_SIZE)
    b_ptrs = B + offs_bk[None, :] * stride_bk + offs_bn[:, None] * stride_bn
    b = tl.load(b_ptrs, mask=offs_bk[None, :] < K, other=0.0)

    # Compute the dot product
    c = tl.dot(a, b)

    # Store the result
    offs_cm = off_cm + tl.arange(0, BLOCK_SIZE)
    offs_cn = off_cn + tl.arange(0, BLOCK_SIZE)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm < M) & (offs_cn < N))


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
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Determine the block size
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    num_block_m = tl.cdiv(M, BLOCK_SIZE)
    num_block_n = tl.cdiv(N, BLOCK_SIZE)
    grid = (num_block_m * num_block_n,)

    # Launch the Triton kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Instead of using torch.matmul, use our Triton-based matrix multiplication
        return triton_matmul(A, B)