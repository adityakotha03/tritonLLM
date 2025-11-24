import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import triton
import triton.language as tl


@triton.jit
def matmul_softmax_kernel(
    x_ptr,          # pointer to input x (BN x D)
    w_ptr,          # pointer to clusters weight (D x (K+G))
    out_ptr,        # pointer to output assignment (BN x (K+G))
    bias_ptr,       # pointer to BN bias (K+G,)
    mean_ptr,       # pointer to BN running mean (K+G,)
    invstd_ptr,     # pointer to BN invstd (K+G,)
    n_rows,         # total number of rows in x (BN)
    n_cols,         # feature dimension D
    n_clusters,     # K + G
    block_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KG: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_k = tl.cdiv(n_cols, BLOCK_D)
    kb = pid % num_blocks_k
    rb = pid // num_blocks_k

    row = rb
    if row >= n_rows:
        return

    offset_d = tl.arange(0, BLOCK_D)
    offset_kg = tl.arange(0, BLOCK_KG)
    mask_d = offset_d < n_cols
    mask_kg = offset_kg < n_clusters

    x_row = x_ptr + row * n_cols + offset_d
    x = tl.load(x_row, mask=mask_d, other=0.0)

    acc = tl.zeros((BLOCK_KG,), dtype=tl.float32)
    for k in range(0, tl.cdiv(n_cols, BLOCK_D)):
        w = tl.load(w_ptr + k * BLOCK_D * n_clusters + offset_d[:, None] * n_clusters + offset_kg[None, :], mask=mask_d[:, None] & mask_kg[None, :], other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)
    
    # Add bias (batch norm bias is fused into linear)
    b = tl.load(bias_ptr + offset_kg, mask=mask_kg, other=0.0)
    acc += b

    # Batch norm: (acc - mean) * invstd
    mean = tl.load(mean_ptr + offset_kg, mask=mask_kg, other=0.0)
    invstd = tl.load(invstd_ptr + offset_kg, mask=mask_kg, other=0.0)
    acc = (acc - mean) * invstd

    # Softmax
    acc = acc - tl.max(acc, axis=0)
    acc = tl.exp(acc)
    sum_acc = tl.sum(acc, axis=0)
    acc = acc / (sum_acc + 1e-10)

    out_row = out_ptr + row * n_clusters + offset_kg
    tl.store(out_row, acc, mask=mask_kg)


@triton.jit
def vlad_kernel(
    assignment_ptr,     # B x K x N (already transposed)
    x_ptr,              # B x N x D
    clusters2_ptr,      # 1 x D x K
    a_sum_ptr,          # B x 1 x K
    out_ptr,            # B x D x K (output vlad)
    B, N, D, K,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (tl.cdiv(D, BLOCK_N))
    d = pid % (tl.cdiv(D, BLOCK_N))

    if b >= B:
        return

    offset_n = tl.arange(0, BLOCK_N)
    offset_d = d * BLOCK_N + offset_n
    mask_d = offset_d < D
    mask_n = offset_n < N

    # Load clusters2 and a_sum: 1 x D x K -> D x K
    clusters2 = tl.load(clusters2_ptr + offset_d[:, None] * K + tl.arange(0, K)[None, :], mask=mask_d[:, None], other=0.0)
    a_sum = tl.load(a_sum_ptr + b * K + tl.arange(0, K), mask=None, other=0.0)  # B x 1 x K -> B x K
    a = clusters2 * a_sum[None, :]  # D x K

    # Compute vlad[:, d, :] = sum_n( assignment[:, :, n] * x[b, n, d] )
    vlad = tl.zeros((K,), dtype=tl.float32)
    for n in range(0, N):
        x_val = tl.load(x_ptr + b * N * D + n * D + offset_d, mask=mask_d, other=0.0)
        assignment_col = tl.load(assignment_ptr + b * K * N + tl.arange(0, K) * N + n, mask=None, other=0.0)  # K,
        vlad += assignment_col * x_val[:, None]  # D x K -> we reduce over n

    vlad = vlad - a
    vlad = vlad / (tl.sqrt(tl.sum(vlad * vlad, axis=0)) + 1e-10)  # L2 normalize per D x K

    # Store output
    out_offset = b * D * K + offset_d[:, None] * K + tl.arange(0, K)[None, :]
    tl.store(out_offset, vlad[None, :], mask=mask_d[:, None])


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * self.feature_size

        # Precompute BN fused weights
        self.register_buffer('bn_bias', None)
        self.register_buffer('bn_weight', None)
        self._update_bn_buffers()

    def _update_bn_buffers(self):
        # Fuse batch norm parameters: y = (x + bias) * invstd + ( - mean * invstd )
        weight = self.batch_norm.weight
        bias = self.batch_norm.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        eps = self.batch_norm.eps
        invstd = 1 / th.sqrt(running_var + eps)
        self.bn_weight = weight * invstd
        self.bn_bias = bias * invstd - running_mean * invstd

    def forward(self, x, mask=None):
        max_sample = x.size(1)
        x_flat = x.view(-1, self.feature_size)  # BN x D
        B, N, D = x.shape
        K = self.cluster_size
        G = self.ghost_clusters
        KG = K + G

        # Update BN buffers if needed
        if self.training:
            self._update_bn_buffers()

        # Allocate output for assignment (BN x KG)
        assignment = torch.empty(x_flat.size(0), KG, dtype=torch.float32, device=x.device)

        # Launch fused matmul + batchnorm + softmax kernel
        def grid_matmul(args):
            return (triton.cdiv(x_flat.size(0), 1), triton.cdiv(D, args['BLOCK_D']) * triton.cdiv(KG, args['BLOCK_KG']))

        matmul_softmax_kernel[grid_matmul](
            x_flat, self.clusters, assignment,
            self.bn_bias, self.batch_norm.running_mean, 1 / th.sqrt(self.batch_norm.running_var + 1e-5),
            x_flat.size(0), D, KG,
            BLOCK_D=32, BLOCK_KG=32, num_warps=4
        )

        # Remove ghost clusters
        assignment = assignment[:, :K]  # BN x K
        assignment = assignment.view(B, N, K)  # B x N x K

        # Sum over N: B x 1 x K
        a_sum = assignment.sum(dim=1, keepdim=True)

        # Transpose assignment to B x K x N for matmul
        assignment = assignment.transpose(1, 2)  # B x K x N

        # Allocate output for vlad: B x D x K
        vlad = torch.empty(B, D, K, dtype=torch.float32, device=x.device)

        # Launch vlad kernel (fused assignment @ x, subtract a, L2 normalize)
        grid_vlad = lambda meta: (B * triton.cdiv(D, meta['BLOCK_N']),)
        vlad_kernel[grid_vlad](
            assignment, x, self.clusters2, a_sum, vlad,
            B, N, D, K, BLOCK_N=32, num_warps=4
        )

        # Flatten and normalize
        vlad = vlad.reshape(B, -1)  # B x (D*K)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad