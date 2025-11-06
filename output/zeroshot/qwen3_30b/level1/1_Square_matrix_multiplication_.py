import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Define program ID and offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current tile
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for the tile of output C
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create mask for output
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load A and B tiles into shared memory
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over tiles of K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offsets for current K tile
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load A tile
        a = tl.load(
            a_ptr + (block_start_m * stride_am + offs_k[:, None] * stride_ak),
            mask=(offs_k[:, None] < K) & (offs_m[None, :] < M),
            other=0.0,
        )

        # Load B tile
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + block_start_n * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Convert to bf16 for Tensor Cores (if supported)
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)

        # Perform matrix multiplication using Tensor Cores
        # tl.dot is optimized for Tensor Cores
        accumulator += tl.dot(a, b)

    # Convert accumulator back to float32
    accumulator = accumulator.to(tl.float32)

    # Store result in output C
    c = accumulator
    tl.store(
        c_ptr + (block_start_m * stride_cm + block_start_n * stride_cn),
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Matrix dimensions must be compatible."

    # Create output tensor
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)

    # Calculate strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Define grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )

    return c


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


N = 2048 * 2