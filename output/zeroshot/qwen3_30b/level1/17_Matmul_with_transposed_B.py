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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    
    # Offset in the output matrix
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for A and B
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create mask for valid indices
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Load A and B blocks
    A = tl.load(
        A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
        mask=(mask_m[:, None] & mask_k[None, :]),
        other=0.0
    )
    B = tl.load(
        B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
        mask=(mask_k[:, None] & mask_n[None, :]),
        other=0.0
    )

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B
        A = tl.load(
            A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        )
        B = tl.load(
            B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0
        )

        # Update accumulator
        accumulator += tl.dot(A, B)

        # Increment k offset
        offs_k += BLOCK_SIZE_K

    # Store output
    C = tl.load(
        C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        mask=(mask_m[:, None] & mask_n[None, :]),
        other=0.0
    )
    tl.store(
        C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        accumulator,
        mask=(mask_m[:, None] & mask_n[None, :])
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton-based matrix multiplication: C = A @ B.T
    Optimized using Tensor Cores and shared memory via tiling.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    N, _ = B.shape

    # Allocate output
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Compute strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Determine grid
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)