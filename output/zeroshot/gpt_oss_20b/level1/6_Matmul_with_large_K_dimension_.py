import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel for matrix multiplication (C = A @ B)
# This kernel is adapted from the Triton matmul examples and
# is tuned for small M,N and a very large K dimension.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    stride_a_m: tl.constexpr, stride_a_k: tl.constexpr,
    stride_b_k: tl.constexpr, stride_b_n: tl.constexpr,
    stride_c_m: tl.constexpr, stride_c_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row and column offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_SIZE_K
    for offs_k in range(0, K, BLOCK_SIZE_K):
        # Load a tile of A (M x K_tile)
        a_ptrs = A_ptr + offs_m[:, None] * stride_a_m + (offs_k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_a_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((offs_k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K), other=0.0)

        # Load a tile of B (K_tile x N)
        b_ptrs = B_ptr + (offs_k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
        b = tl.load(b_ptrs, mask=((offs_k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Compute partial product and accumulate
        acc += tl.dot(a, b)

    # Write the result to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using a custom Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Inner dimensions must match."

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "K dimension mismatch."

    # Convert inputs to fp16 if possible
    dtype = torch.float16
    A_fp16 = A.to(dtype)
    B_fp16 = B.to(dtype)

    # Prepare output tensor
    C = torch.empty((M, N), dtype=dtype, device=A.device)

    # Strides
    stride_a_m, stride_a_k = A_fp16.stride()
    stride_b_k, stride_b_n = B_fp16.stride()
    stride_c_m, stride_c_n = C.stride()

    # Grid dimensions
    grid = (
        triton.cdiv(M, 128),
        triton.cdiv(N, 128),
    )

    # Launch the kernel
    matmul_kernel[grid](
        A_fp16.data_ptr(), B_fp16.data_ptr(), C.data_ptr(),
        M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        stride_a_m=stride_a_m, stride_a_k=stride_a_k,
        stride_b_k=stride_b_k, stride_b_n=stride_b_n,
        stride_c_m=stride_c_m, stride_c_n=stride_c_n,
    )

    return C.to(A.dtype)  # convert back to original dtype if needed


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)