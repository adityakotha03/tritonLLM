import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a_m, stride_a_n,
    stride_b_m, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Matrix multiplication kernel: C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    All tensors are in fp16; accumulators are fp32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_start = k
        # Compute offset for block of A (BLOCK_SIZE_M x BLOCK_SIZE_K)
        a_offsets = (
            start_m[:, None] * stride_a_m
            + (k_start + tl.arange(0, BLOCK_SIZE_K)) * stride_a_n
        )
        # Compute offset for block of B (BLOCK_SIZE_K x BLOCK_SIZE_N)
        b_offsets = (
            (k_start + tl.arange(0, BLOCK_SIZE_K)) * stride_b_m
            + start_n[None, :] * stride_b_n
        )

        # Load tiles with masking for boundaries
        a = tl.load(
            A_ptr + a_offsets,
            mask=(start_m[:, None] < M) & (k_start + tl.arange(0, BLOCK_SIZE_K) < K),
            other=0.0,
            dtype=tl.float16,
        )
        b = tl.load(
            B_ptr + b_offsets,
            mask=(k_start + tl.arange(0, BLOCK_SIZE_K) < K) & (start_n[None, :] < N),
            other=0.0,
            dtype=tl.float16,
        )

        acc += tl.dot(a, b)

    # Store the result
    c_offsets = (
        start_m[:, None] * stride_c_m
        + start_n[None, :] * stride_c_n
    )
    tl.store(
        C_ptr + c_offsets,
        acc,
        mask=(start_m[:, None] < M) & (start_n[None, :] < N),
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using Triton custom kernel.
    Input tensors must be on CUDA and 2‑D.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 2 and B.dim() == 2, "Only 2‑D matrices are supported."
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."

    # Convert to FP16 for Tensor Core usage
    A_fp16 = A.to(torch.float16)
    B_fp16 = B.to(torch.float16)
    C_fp16 = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Block sizes (tuned for 80 GB A100)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    # Grid dimensions
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    matmul_kernel[grid](
        A_fp16,
        B_fp16,
        C_fp16,
        M,
        N,
        K,
        A_fp16.stride(0),
        A_fp16.stride(1),
        B_fp16.stride(0),
        B_fp16.stride(1),
        C_fp16.stride(0),
        C_fp16.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return C_fp16.to(torch.float32)


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.matmul with a custom Triton kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)