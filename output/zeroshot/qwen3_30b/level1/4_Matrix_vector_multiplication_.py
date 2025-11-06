import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Map program IDs to block indices
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = tl.program_id(1) * BLOCK_SIZE_N

    # Initialize offsets for current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create mask to prevent out-of-bounds access
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load A and B blocks
    offs_am = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    offs_bk = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    A = tl.load(A_ptr + offs_am, mask=(offs_k[None, :] < K), other=0.0)
    B = tl.load(B_ptr + offs_bk, mask=(offs_k[:, None] < K), other=0.0)

    # Accumulator for C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B tiles
        A = tl.load(
            A_ptr + offs_am,
            mask=(offs_k[None, :] < K) & (k * BLOCK_SIZE_K + offs_k[None, :] < K),
            other=0.0
        )
        B = tl.load(
            B_ptr + offs_bk,
            mask=(offs_k[:, None] < K) & (k * BLOCK_SIZE_K + offs_k[:, None] < K),
            other=0.0
        )

        # Perform matrix multiplication using tensor cores (FP16/BF16)
        accumulator += tl.dot(A, B)

        # Advance offset
        offs_am += BLOCK_SIZE_K * stride_ak
        offs_bk += BLOCK_SIZE_K * stride_bk

    # Convert to output type
    C = accumulator.to(tl.float16)

    # Store result with masking
    offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptr + offs_cm, C, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on GPU and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
        A = A.contiguous()
        B = B.contiguous()

        # Extract shapes
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Matrix dimensions mismatch."

        # Define block sizes and tuning parameters
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8

        # Ensure we use tensor cores
        assert A.dtype == torch.float16 or A.dtype == torch.bfloat16, "Use fp16 or bf16 for tensor core utilization."

        # Calculate strides
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = (M, 1) if N == 1 else (M, N)

        # Prepare output tensor
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)

        # Grid setup: one block per M/N tile
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

        # Launch Triton kernel with fusion (matmul only in this case)
        matmul_kernel[grid](
            A, B, C,
            M, K, N,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            ACTIVATION="none"
        )

        return C