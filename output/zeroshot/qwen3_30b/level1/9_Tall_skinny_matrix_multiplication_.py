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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Calculate the program ID along the M dimension
    pid = tl.program_id(0)
    pid_m = pid // (BLOCK_SIZE_N // BLOCK_SIZE_N)
    pid_n = pid % (BLOCK_SIZE_N // BLOCK_SIZE_N)
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for the block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create mask to handle padding
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load A and B
    a = tl.load(
        A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=mask_m[:, None] & (offs_k[None, :] < K),
        other=0.0
    )
    b = tl.load(
        B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(offs_k[:, None] < K) & mask_n[None, :],
        other=0.0
    )

    # Perform matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator += tl.dot(a, b, allow_tf32=True)
    
    # Apply activation if specified
    if ACTIVATION == "relu":
        accumulator = tl.max(accumulator, 0)

    # Store output
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        accumulator,
        mask=mask
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor, activation=None):
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Input tensor shapes
    M, K = A.shape
    K, N = B.shape

    # Prepare output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define block sizes and grid size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    # Determine number of blocks
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks = num_blocks_m * num_blocks_n

    # Grid configuration
    grid = lambda meta: (num_blocks,)

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
        ACTIVATION=activation
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B, activation=None)