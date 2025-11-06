import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_a_n, stride_a_m, stride_a_k,
    stride_b_k, stride_b_l,
    stride_c_n, stride_c_m, stride_c_l,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # Define program ID and block offsets
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Calculate block offsets
    block_n = pid_n * BLOCK_SIZE_N
    block_m = pid_m * BLOCK_SIZE_M
    block_k = pid_k * TILE_K

    # Create offsets for the block
    offs_n = block_n + tl.arange(0, BLOCK_SIZE_N)
    offs_m = block_m + tl.arange(0, BLOCK_SIZE_M)
    offs_k = block_k + tl.arange(0, TILE_K)

    # Create masks to handle out-of-bounds indices
    mask_n = offs_n < N
    mask_m = offs_m < M
    mask_k = offs_k < K

    # Load A (N, M, K) and B (K, L) tiles
    # A: (BLOCK_SIZE_N, BLOCK_SIZE_M, TILE_K)
    A_ptrs = A_ptr + (offs_n[:, None] * stride_a_n + offs_m[None, :] * stride_a_m + offs_k[None, :] * stride_a_k)
    A = tl.load(A_ptrs, mask=(mask_n[:, None] & mask_m[None, :] & mask_k[None, :]), other=0.0)

    # B: (TILE_K, BLOCK_SIZE_L)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_b_k + offs_m[None, :] * stride_b_l)
    B = tl.load(B_ptrs, mask=(mask_k[:, None] & mask_m[None, :]), other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_L), dtype=tl.float32)

    # Perform matrix multiplication
    for _ in range(0, K, TILE_K):
        # Load A tile
        A = tl.load(
            A_ptr + (offs_n[:, None] * stride_a_n + offs_m[None, :] * stride_a_m + offs_k[None, :] * stride_a_k),
            mask=(mask_n[:, None] & mask_m[None, :] & mask_k[None, :]),
            other=0.0
        )
        # Load B tile
        B = tl.load(
            B_ptr + (offs_k[:, None] * stride_b_k + offs_m[None, :] * stride_b_l),
            mask=(mask_k[:, None] & mask_m[None, :]),
            other=0.0
        )

        # Update accumulator
        acc += tl.dot(A, B)

        # Update k offset
        offs_k += TILE_K
        mask_k = offs_k < K

    # Store output
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    mask_l = offs_l < L
    C_ptrs = C_ptr + (offs_n[:, None] * stride_c_n + offs_m[None, :] * stride_c_m + offs_l[None, :] * stride_c_l)
    tl.store(C_ptrs, acc, mask=(mask_n[:, None] & mask_m[None, :] & mask_l[None, :]))


def triton_matmul(A, B):
    """
    Perform 3D tensor-matrix multiplication using Triton kernel.
    Input A: (N, M, K), B: (K, L) -> Output: (N, M, L)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()

    # Shapes
    N, M, K = A.shape
    L = B.shape[1]

    # Strides
    stride_a_n, stride_a_m, stride_a_k = A.stride()
    stride_b_k, stride_b_l = B.stride()
    stride_c_n, stride_c_m, stride_c_l = N * M, M, 1

    # Define block sizes and tile size
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_L = 128
    TILE_K = 64

    # Grid configuration: (N_blocks, M_blocks, K_blocks)
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (K + TILE_K - 1) // TILE_K

    grid = (grid_n, grid_m, grid_k)

    # Launch kernel
    matmul_kernel[grid](
        A, B, None,
        N, M, K, L,
        stride_a_n, stride_a_m, stride_a_k,
        stride_b_k, stride_b_l,
        stride_c_n, stride_c_m, stride_c_l,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        TILE_K=TILE_K
    )

    # Create output tensor
    C = torch.empty(N, M, L, dtype=A.dtype, device=A.device)
    # Re-launch kernel to write to output
    matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_a_n, stride_a_m, stride_a_k,
        stride_b_k, stride_b_l,
        stride_c_n, stride_c_m, stride_c_l,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        TILE_K=TILE_K
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)