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
    BLOCK: tl.constexpr,
):
    # Each program processes a tile of size BLOCK x BLOCK
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block position in the matrix
    block_m = pid // (N // BLOCK)
    block_n = pid % (N // BLOCK)
    # Compute the block offset
    offs_m = block_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = block_n * BLOCK + tl.arange(0, BLOCK)
    # Load A and B
    a = tl.load(A + offs_m[:, None] * stride_am + tl.arange(0, BLOCK)[None, :] * stride_ak, mask=offs_m[:, None] < M, other=0.0)
    b = tl.load(B + tl.arange(0, BLOCK)[None, :] * stride_bk + offs_n[None, :] * stride_bn, mask=offs_n[None, :] < N, other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=offs_m[:, None] < M & offs_n[None, :] < N)


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
    K2, N = B.shape
    assert K == K2, "Matrix dimensions do not match."
    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Number of elements in the tensor
    BLOCK = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M * N + meta["BLOCK"] * meta["BLOCK"] - 1) // (meta["BLOCK"] * meta["BLOCK"]),)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, out, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), out.stride(0), out.stride(1), BLOCK=BLOCK)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.matmul(A, B)", call our Triton-based matrix multiplication
        return triton_matmul(A, B)