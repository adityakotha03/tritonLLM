import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------
# Triton kernels
# ---------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 512, "BLOCK_K": 8}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A = tl.load(
            A_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_am
            + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
            mask=(row_start + tl.arange(0, BLOCK_M))[:, None] < M
            & (k + tl.arange(0, BLOCK_K))[None, :] < K,
            other=0.0,
        )
        B = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk
            + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_bn,
            mask=(k + tl.arange(0, BLOCK_K))[:, None] < K
            & (col_start + tl.arange(0, BLOCK_N))[None, :] < N,
            other=0.0,
        )
        acc += tl.dot(A, B)

    C = acc.to(tl.float32)

    tl.store(
        C_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_cm
        + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_cn,
        C,
        mask=(row_start + tl.arange(0, BLOCK_M))[:, None] < M
        & (col_start + tl.arange(0, BLOCK_N))[None, :] < N,
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of matrix multiplication A @ B.
    Supports only float32 tensors.
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am=1,
        stride_ak=M,
        stride_bk=1,
        stride_bn=K,
        stride_cm=1,
        stride_cn=N,
    )
    return C


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["n"],
)
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # subtract max for numerical stability
    max_val = tl.max(x, mask=mask)
    x = x - max_val

    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, mask=mask)

    out = exp_x / sum_exp
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Triton implementation of softmax along a single dimension.
    Only supports dim=1 for 2D tensors.
    """
    assert x.is_cuda
    if dim != 1:
        raise NotImplementedError("Only dim=1 supported in this Triton kernel")
    B, K = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: ((B * K + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softmax_kernel[grid](x.flatten(), out.flatten(), B * K)
    return out


# ---------------------------------------------
# Optimized Model
# ---------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters, device="cuda"))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size, device="cuda"))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x: B x N x D
        """
        B, N, D = x.shape
        max_sample = N

        # reshape for matmul
        x_flat = x.reshape(B * N, D)  # (BN) x D

        # Assignment logits: (BN x D) @ (D x K+G) -> (BN x K+G)
        assignment = triton_matmul(x_flat, self.clusters)

        # BatchNorm
        assignment = self.batch_norm(assignment)

        # Softmax
        assignment = triton_softmax(assignment, dim=1)

        # remove ghost clusters
        assignment = assignment[:, :self.cluster_size]  # (BN x K)

        # reshape back
        assignment = assignment.view(B, N, self.cluster_size)  # B x N x K
        a_sum = assignment.sum(dim=1, keepdim=True)  # B x 1 x K
        a = a_sum * self.clusters2  # B x 1 x D x K -> broadcast

        # transpose for VLAD
        assignment_T = assignment.transpose(1, 2)  # B x K x N

        # Reshape x for second matmul
        x_reshaped = x_flat.view(B, N, D)  # B x N x D

        # VLAD accumulation: (B x K x N) @ (B x N x D) -> B x K x D
        vlad = triton_matmul(assignment_T.reshape(-1, N), x_reshaped.reshape(-1, D))
        vlad = vlad.view(B, self.cluster_size, D)

        # Subtract residual
        vlad = vlad - a

        # Intra-normalization
        vlad = F.normalize(vlad, dim=2, p=2)

        # Flatten and final L2
        vlad = vlad.reshape(B, self.out_dim)
        vlad = F.normalize(vlad, dim=1, p=2)

        return vlad