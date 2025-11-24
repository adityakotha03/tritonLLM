import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_features, out_features,
    stride_xb, stride_xn, stride_xd,
    stride_wd, stride_wk,
    stride_outb, stride_outn, stride_outk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + pid_b * stride_xb + offs_m[:, None] * stride_xn + offs_d[None, :] * stride_xd
    w_ptrs = w_ptr + offs_d[:, None] * stride_wd + offs_k[None, :] * stride_wk
    out_ptrs = out_ptr + pid_b * stride_outb + offs_m[:, None] * stride_outn + offs_k[None, :] * stride_outk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d in range(0, in_features, BLOCK_K):
        mask_d = (d + offs_d) < in_features
        x = tl.load(x_ptrs, mask=mask_d[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_d[:, None], other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xd
        w_ptrs += BLOCK_K * stride_wd

    out = accumulator.to(tl.float16)
    mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < batch_size
    mask_k = (pid_k * BLOCK_N + tl.arange(0, BLOCK_N)) < out_features
    mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    batch_size, n_rows, n_cols,
    stride_inputb, stride_inputr, stride_inputc,
    stride_outputb, stride_outputr, stride_outputc,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    row_start = pid_r * BLOCK_R
    rows = row_start + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)

    mask_rows = rows < n_rows
    mask_cols = cols < n_cols
    mask = mask_rows[:, None] & mask_cols[None, :]

    offsets = (pid_b * stride_inputb + rows[:, None] * stride_inputr + cols[None, :] * stride_inputc)
    input_ptrs = input_ptr + offsets
    output_ptrs = output_ptr + offsets

    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask_load = row_mask[:, None] & col_mask[None, :]

    x = tl.load(input_ptrs, mask=mask_load, other=-float('inf'))
    x = x - tl.max(x, axis=1)[:, None]
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=1)[:, None]
    softmax_output = exp_x / (sum_exp + 1e-10)

    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def batch_matmul_trans_b_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = (k + offs_k) < K
        mask_m = offs_m < M
        mask_n = offs_n < N
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def l2_normalize_kernel(
    x_ptr, out_ptr,
    N, D,
    stride_xn, stride_xd,
    stride_outn, stride_outd,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < D
    mask = mask_n[:, None] & mask_d[None, :]

    offsets = offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
    x_ptrs = x_ptr + offsets
    out_ptrs = out_ptr + offsets

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_sq = x * x
    norm_sq = tl.sum(x_sq, axis=1)
    norm = tl.sqrt(norm_sq + 1e-10)
    x_norm = x / norm[:, None]

    tl.store(out_ptrs, x_norm, mask=mask)


def triton_batched_matmul(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
            triton.cdiv(K, META['BLOCK_K'])
        )

    batched_matmul_kernel[grid](
        a, b, c,
        M, K, N,
        0, 1, K,
        1, K,
        0, 1, N,
        BLOCK_M=64, BLOCK_N=32, BLOCK_K=32
    )
    return c


def triton_softmax(x):
    assert x.is_cuda
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    out = torch.empty_like(x)

    def grid(META):
        return (x.shape[0], triton.cdiv(x.shape[1], META['BLOCK_R']))

    softmax_kernel[grid](
        x, out,
        x.shape[0], x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_R=32, BLOCK_C=128
    )
    return out


def triton_batch_matmul_trans_b(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    batch_matmul_trans_b_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return c


def triton_l2_normalize(x, dim=-1):
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    shape_orig = x.shape
    x = x.reshape(-1, shape_orig[-1])
    out = torch.empty_like(x)

    def grid(META):
        return (triton.cdiv(x.shape[0], META['BLOCK_N']),)

    l2_normalize_kernel[grid](
        x, out,
        x.shape[0], x.shape[1],
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=32, BLOCK_D=512
    )

    out = out.reshape(shape_orig)
    if dim != -1 and dim != x.ndim - 1:
        out = out.transpose(dim, -1)
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
        self.out_dim = self.cluster_size * self.feature_size

    def forward(self, x, mask=None):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = triton_batched_matmul(x, self.clusters)
        assignment = self.batch_norm(assignment)
        assignment = triton_softmax(assignment.view(-1, self.cluster_size + self.ghost_clusters))
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)
        a_sum = th.sum(assignment, dim=1, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)
        x_restore = x.view(-1, max_sample, self.feature_size)
        vlad = triton_batch_matmul_trans_b(assignment, x_restore)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a
        vlad = triton_l2_normalize(vlad, dim=1)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = triton_l2_normalize(vlad, dim=1)
        return vlad