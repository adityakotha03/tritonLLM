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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Initialize program ID and block offsets
    pid = tl.program_id(0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    
    # Offset for blocks
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for the block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute mask for boundary conditions
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load A and B in blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B with masking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)

        # Perform matrix multiplication
        accumulator += tl.dot(a, b)

        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation (ReLU)
    if ACTIVATION == "RELU":
        accumulator = tl.maximum(accumulator, 0.0)

    # Convert to output type (e.g., float16/bfloat16)
    accumulator = accumulator.to(tl.float16)

    # Store output
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, activation="RELU"):
    """
    Perform matrix multiplication with Triton kernel.
    Supports fused ReLU activation.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape[1] == B.shape[0], "Incompatible shapes for matmul"

    # Make sure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Get dimensions
    M, K = A.shape
    K, N = B.shape

    # Allocate output
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Define block sizes and tuning parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Calculate number of blocks
    num_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n,)

    # Get strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
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
        return triton_matmul(A, B, activation="RELU")