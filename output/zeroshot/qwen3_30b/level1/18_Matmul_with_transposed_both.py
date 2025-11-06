import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,  # Strides for A (M, K)
    stride_bk, stride_bn,  # Strides for B (K, N)
    stride_cm, stride_cn,  # Strides for C (M, N)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Matrix multiplication: C = A @ B
    # A is (M, K), B is (K, N), C is (M, N)

    # Program ID for this block
    pid = tl.program_id(0)
    # Calculate block row and column indices
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = tl.program_id(1) * BLOCK_SIZE_N

    # Offset for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute current K range
        k_start = k * BLOCK_SIZE_K
        k_end = min(k_start + BLOCK_SIZE_K, K)
        k_mask = offs_k < k_end

        # Load A and B for current K block
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Load A and B, with masking to handle edge cases
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # Perform matmul with accumulation
        accumulator += tl.dot(a, b, allow_tf32=True)

    # Apply activation (ReLU is fused here)
    if ACTIVATION == "RELU":
        accumulator = tl.maximum(accumulator, 0.0)

    # Clamp and cast to output type (BFloat16 or Float16)
    accumulator = tl.clip(accumulator, -65504.0, 65504.0)

    # Write output
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, activation: str = "RELU"):
    """
    Wrapper function to launch the Triton kernel for matmul with fused ReLU.
    Uses FP16/BF16 with Tensor Core optimization on A100.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape[1] == B.shape[0], "Shape mismatch for matmul: A.shape[1] != B.shape[0]"
    
    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()

    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Mismatch in inner dimensions"

    # Output tensor
    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    # Configurable constants
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    # Grid setup
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

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
        ACTIVATION=activation,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass using Triton kernel for fused matmul + ReLU.
        Transposes A and B to match C = A.T @ B.T, but computes A @ B in original layout.
        """
        # We want C = A.T @ B.T -> equivalent to (B @ A).T
        # But in Triton, we compute B @ A directly (since it's more efficient to transpose on GPU)
        # So we compute D = B @ A, then transpose for output
        D = triton_matmul(A.T, B.T, activation="RELU")
        return D.T