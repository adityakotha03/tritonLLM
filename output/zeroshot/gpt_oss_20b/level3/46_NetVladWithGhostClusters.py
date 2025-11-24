import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import torch as th

# Triton kernel for matrix multiplication: C = A @ B
# A: M x K, B: K x N, C: M x N
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + (row_start[:, None] * stride_am + k[None, :] * stride_ak),
            mask=(row_start[:, None] < M) & (k[None, :] < K),
        )
        b = tl.load(
            B_ptr + (k[:, None] * stride_bk + col_start[None, :] * stride_bn),
            mask=(k[:, None] < K) & (col_start[None, :] < N),
        )
        acc += tl.dot(a, b)

    if row_start < M and col_start < N:
        tl.store(
            C_ptr + (row_start[:, None] * stride_cm + col_start[None, :] * stride_cn),
            acc,
            mask=(row_start[:, None] < M) & (col_start[None, :] < N),
        )

def triton_matmul(A: th.Tensor, B: th.Tensor) -> th.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = th.empty((M, N), dtype=A.dtype, device=A.device)
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
        stride_am=1, stride_ak=1,
        stride_bk=K, stride_bn=1,
        stride_cm=1, stride_cn=N,
    )
    return C

# Triton kernel for softmax along last dimension (dim=1)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _softmax_kernel(
    inp_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    # compute max
    x_max = tl.max(x, axis=0)
    # subtract max and exponentiate
    x_exp = tl.exp(x - x_max)
    # compute sum
    x_sum = tl.sum(x_exp, axis=0)
    # normalize
    out = x_exp / x_sum
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softmax(x: th.Tensor, dim: int = 1) -> th.Tensor:
    # assume dim=1 for 2D matrix
    n = x.size(dim)
    out = th.empty_like(x)
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _softmax_kernel[grid](x, out, n, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x: th.Tensor, mask=None) -> th.Tensor:
        max_sample = x.size(1)
        # reshape B N D -> BN D
        x_flat = x.view(-1, self.feature_size)

        # assignment = x_flat @ clusters
        assignment = triton_matmul(x_flat, self.clusters)

        assignment = self.batch_norm(assignment)

        assignment = triton_softmax(assignment, dim=1)

        # remove ghost clusters
        assignment = assignment[:, :self.cluster_size]

        # reshape to B N K
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = assignment.sum(dim=1, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B K N

        x_reshaped = x_flat.view(-1, max_sample, self.feature_size)

        # vlad = assignment @ x_reshaped
        vlad = triton_matmul(assignment, x_reshaped)

        vlad = vlad.transpose(1, 2)  # B D K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad, dim=2)

        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, dim=1)
        return vlad