import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    b, i, j, l, k,
    stride_Ab, stride_Ai, stride_Aj, stride_Al,
    stride_Bl, stride_Bk,
    stride_Cb, stride_Ci, stride_Cj, stride_Ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    pid_k_block = tl.program_id(3)

    # Offsets for j and k blocks
    range_j = pid_j * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    range_k = (pid_k_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    # Mask for j
    mask_j = range_j < j
    mask_k = range_k < k

    # Base offsets for A and C
    A_block_base = pid_b * stride_Ab + pid_i * stride_Ai + range_j[:, None] * stride_Aj + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_Al
    C_block_base = pid_b * stride_Cb + pid_i * stride_Ci + range_j[:, None] * stride_Cj + range_k[None, :] * stride_Ck

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over l dimension in blocks
    for lb in range(0, l, BLOCK_SIZE_K):
        # Compute mask for k dimension
        A_ptrs = A_ptr + A_block_base + lb
        B_ptrs = B_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_Bl + range_k[None, :] * stride_Bk

        # Load A and B tiles
        a = tl.load(A_ptrs, mask=mask_j[:, None] & ((lb + tl.arange(0, BLOCK_SIZE_K)[None, :]) < l), other=0.0)
        b = tl.load(B_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < l - lb) & mask_k[None, :], other=0.0)

        # Matmul in float32
        acc += tl.dot(a, b)

        # Update A block base for next K block
        A_block_base += BLOCK_SIZE_K * stride_Al

    # Store result
    C_ptrs = C_ptr + C_block_base
    mask = mask_j[:, None] & mask_k[None, :]
    tl.store(C_ptrs, acc, mask=mask)


def triton_matmul_4d_2d(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU."
    assert A.shape[-1] == B.shape[0], "Inner dimension mismatch."

    b, i, j, l = A.shape
    k = B.shape[1]

    # Output shape: (b, i, j, k)
    C = torch.empty((b, i, j, k), device=A.device, dtype=A.dtype)

    # Constants
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Grid: (b, i, j_blocks, k_blocks)
    grid = (b, i, triton.cdiv(j, BLOCK_SIZE_M), triton.cdiv(k, BLOCK_SIZE_N))

    # Launch kernel
    matmul_4d_2d_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        return triton_matmul_4d_2d(A, B)