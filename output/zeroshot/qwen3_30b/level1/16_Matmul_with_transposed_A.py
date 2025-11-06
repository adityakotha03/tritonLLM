import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Calculate program ID and block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Calculate block offsets
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    block_start_k = pid_k * BLOCK_K

    # Create offsets for the block
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_K)

    # Create mask for valid indices
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_n[None, :] & mask_k[None, :]

    # Load A and B blocks
    a = tl.load(
        a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
        mask=mask,
        other=0.0
    )
    b = tl.load(
        b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
        mask=mask,
        other=0.0
    )

    # Perform dot product with accumulation
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks
        a_block = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + (block_start_k + i * BLOCK_K) * stride_ak),
            mask=mask_m[:, None] & (offs_k[None, :] < K),
            other=0.0
        )
        b_block = tl.load(
            b_ptr + ((block_start_k + i * BLOCK_K) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[None, :] < K) & mask_n[None, :],
            other=0.0
        )
        # Perform matrix multiplication
        accumulator += tl.dot(a_block, b_block)

    # Apply activation (none in this case)
    if ACTIVATION == "none":
        pass
    elif ACTIVATION == "relu":
        accumulator = tl.relu(accumulator)

    # Store result
    c = accumulator.to(tl.float16)
    tl.store(
        c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor, activation="none"):
    """
    Perform matrix multiplication with Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape[1] == B.shape[0], "Invalid dimensions for matrix multiplication"
    assert A.dtype == B.dtype, "Input tensors must have the same dtype"

    # Ensure contiguous layout
    A = A.contiguous()
    B = B.contiguous()

    # Get dimensions
    M, K = A.shape
    K, N = B.shape

    # Prepare output
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Define block sizes and grid
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    GROUP_SIZE_M = 8
    GRID = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    # Get strides
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Launch kernel
    matmul_kernel[GRID](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Perform A.T @ B using optimized Triton kernel
        return triton_matmul(A.T, B, activation="none")