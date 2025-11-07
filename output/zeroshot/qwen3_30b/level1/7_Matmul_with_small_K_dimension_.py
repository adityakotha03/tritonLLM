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
    ACTIVATION: tl.constexpr,
):
    # Matrix multiplication: C = A @ B
    # A is (M, K), B is (K, N), C is (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets for A and B
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    block_start_k = pid_k * BLOCK_K

    # Create offsets for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_K)

    # Create masks to prevent out-of-bounds access
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Load A and B chunks from global memory
    # Load A: (BLOCK_M, BLOCK_K)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # Load B: (BLOCK_K, BLOCK_N)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Accumulate the result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Perform matrix multiplication using tensor cores
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate the result
        accumulator += tl.dot(a, b)

        # Update offsets
        block_start_k += BLOCK_K
        offs_k = block_start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply activation if specified (e.g., ReLU)
    if ACTIVATION == "ReLU":
        accumulator = tl.maximum(accumulator, 0.0)

    # Convert to output type
    c = accumulator.to(tl.float16)

    # Write C: (BLOCK_M, BLOCK_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor, activation: str = None):
    # Ensure inputs are on CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()

    # Get dimensions
    M, K = A.shape
    K, N = B.shape

    # Allocate output tensor
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Define block sizes and group size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8

    # Compute strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Compute grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(K, meta["BLOCK_K"]),
    )

    # Launch the Triton kernel
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
        ACTIVATION=activation,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Replace torch.matmul with custom Triton-optimized version
        return triton_matmul(A, B, activation=None)