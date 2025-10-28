import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1},
            num_warps=8, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'SPLIT_K': 1},
            num_warps=8, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 4},
            num_warps=8, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'SPLIT_K': 8},
            num_warps=4, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'SPLIT_K': 8},
            num_warps=4, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'SPLIT_K': 16},
            num_warps=4, num_stages=4
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_splitk_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * num_pid_n
    group_id = pid_0 // group_size
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_0 % GROUP_M)
    pid_n = (pid_0 % group_size) // GROUP_M

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Determine K range for this split
    k_begin = (pid_k * K) // SPLIT_K
    k_end = ((pid_k + 1) * K) // SPLIT_K

    for k in range(k_begin, k_end, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Cast to BF16 for Tensor Core friendly dot; accumulation remains FP32
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D tensors"
    assert A.shape[1] == B.shape[0], "Inner dimensions must match"

    # Move to GPU and choose BF16 inputs to use A100 Tensor Cores, accumulate in FP32
    device = torch.device('cuda')
    A_bf16 = A.to(device=device, dtype=torch.bfloat16, copy=True).contiguous()
    B_bf16 = B.to(device=device, dtype=torch.bfloat16, copy=True).contiguous()

    M, K = A_bf16.shape
    K2, N = B_bf16.shape
    assert K == K2

    C = torch.zeros((M, N), device=device, dtype=torch.float32)

    stride_am, stride_ak = A_bf16.stride()
    stride_bk, stride_bn = B_bf16.stride()
    stride_cm, stride_cn = C.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
            meta['SPLIT_K'],
        )

    matmul_splitk_kernel[grid](
        A_bf16, B_bf16, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using a custom Triton kernel with split-K and Tensor Core acceleration.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# Keep the same input generation for compatibility
M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []