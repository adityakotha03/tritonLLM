import torch
import triton
import triton.language as tl

# Triton kernel for matrix multiplication
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Offsets for A
        m = block_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
        k_tile = k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        a_offset = m * stride_am + k_tile * stride_an
        a_mask = (m < M) & (k_tile < K)

        # Offsets for B
        n = block_n + tl.arange(0, BLOCK_SIZE_N)[None, :]
        b_offset = k_tile * stride_bm + n * stride_bn
        b_mask = (k_tile < K) & (n < N)

        a = tl.load(a_offset, mask=a_mask, other=0.0)
        b = tl.load(b_offset, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    m = block_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
    n = block_n + tl.arange(0, BLOCK_SIZE_N)[None, :]
    c_offset = m * stride_cm + n * stride_cn
    c_mask = (m < M) & (n < N)
    tl.store(c_offset, acc, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication A @ B using the Triton kernel.
    A: (M, K)
    B: (K, N)
    """
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am=K, stride_an=1,
        stride_bm=N, stride_bn=1,
        stride_cm=N, stride_cn=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(torch.nn.Module):
    """
    Optimized model using Triton kernel for 4D tensor-matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs C[b,i,j,k] = sum_l A[b,i,j,l] * B[l,k]
        """
        b, i, j, l = A.shape
        A_flat = A.reshape(b * i * j, l)
        C_flat = matmul_triton(A_flat, B)
        return C_flat.reshape(b, i, j, B.shape[1])