import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # strides for A
    stride_bk, stride_bn,  # strides for B
    stride_cm, stride_cn,  # strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID and block offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets for the current block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create offsets within the block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Create mask for valid indices
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Load A and B tiles
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Accumulate results in fp32
        acc += tl.dot(a, b, allow_tf32=True)

        # Update pointers for next block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Clamp to valid region
    acc = acc * tl.where(mask_m[:, None], 1.0, 0.0) * tl.where(mask_n[None, :], 1.0, 0.0)

    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication using a custom Triton kernel optimized for large K.
    Leverages tensor cores with BF16 input, fused memory layout, and tiling for large K.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16, "Use BF16 for optimal Tensor Core performance"

    M, K = A.shape
    K, N = B.shape

    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    # Configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64  # Optimized for 131072 * 4 K
    GROUP_SIZE_M = 8

    # Calculate grid dimensions
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid_k = triton.cdiv(K, BLOCK_SIZE_K)

    # Define grid
    grid = (grid_m, grid_n, grid_k)

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel with BF16 and large K optimization
        return triton_matmul(A, B)