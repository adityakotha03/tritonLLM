import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Calculate program id
    pid = tl.program_id(axis=0)
    pid_m = pid // (tl.cdiv(N, BLOCK_N))
    pid_n = pid % (tl.cdiv(N, BLOCK_N))

    # Offset for the block
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # Create offsets
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B
        a = tl.load(A_ptrs, mask=offs_k[None, :] < (k + 1) * BLOCK_K, other=0.0)
        b = tl.load(B_ptrs, mask=offs_k[:, None] < (k + 1) * BLOCK_K, other=0.0)

        # Perform matmul
        acc += tl.dot(a, b)

        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Apply activation (optional, but we can fuse it if needed)
    if ACTIVATION == "none":
        pass
    elif ACTIVATION == "relu":
        acc = tl.where(acc > 0, acc, 0)

    # Scale and store output
    acc = acc.to(tl.float16)

    # Final output mask
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, activation: str = "none") -> torch.Tensor:
    # Ensure inputs are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()

    # Expected shapes
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication"

    # Output tensor
    C = torch.empty(M, N, device=A.device, dtype=torch.float16)

    # Stride info
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    # Grid calculation
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    grid = (num_blocks_m * num_blocks_n,)

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Perform fused matmul with optional ReLU activation, using Triton for high performance
        return triton_matmul(A, B, activation="none")  # Can be "relu" if needed