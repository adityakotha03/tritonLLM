import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Constants
MAX_BLOCK_SIZE = 512
# We'll let Triton autotune this, so we don't hardcode it
# But we'll set reasonable bounds

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row of data
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load row of input
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float('inf'))

    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum
    tl.store(out_ptr + row * n_cols + offsets, x, mask=mask)

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(n, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(n, BLOCK_SIZE_N)

    # Offset calculation
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create mask
    mask_m = offs_m < m
    mask_n = offs_n < n
    mask = mask_m[:, None] & mask_n[None, :]

    # Load weights
    a = tl.load(a_ptr + (offs_m[:, None] * k + offs_k[None, :]), mask=mask_m[:, None])
    b = tl.load(b_ptr + (offs_k[:, None] * n + offs_n[None, :]), mask=mask_n[None, :])

    # Compute dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(a_ptr + (offs_m[:, None] * k + offs_k[None, :]), mask=mask_m[:, None])
        b = tl.load(b_ptr + (offs_k[:, None] * n + offs_n[None, :]), mask=mask_n[None, :])
        acc += tl.dot(a, b)

    # Store result
    out = acc.to(tl.float32)
    tl.store(out_ptr + (offs_m[:, None] * n + offs_n[None, :]), out, mask=mask)


@triton.jit
def weighted_sum_kernel(
    assignments_ptr,
    features_ptr,
    output_ptr,
    batch_size,
    num_features,
    num_clusters,
    feature_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for batch
    batch_id = tl.program_id(0)

    # Each thread handles one element of the output
    for i in range(num_clusters):
        for j in range(feature_size):
            # Calculate output index
            out_idx = batch_id * num_clusters * feature_size + i * feature_size + j

            # Calculate the sum
            sum_val = 0.0
            for n in range(num_features):
                assign_val = tl.load(assignments_ptr + batch_id * num_features * num_clusters + n * num_clusters + i)
                feat_val = tl.load(features_ptr + batch_id * num_features * feature_size + n * feature_size + j)
                sum_val += assign_val * feat_val

            tl.store(output_ptr + out_idx, sum_val)

@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute L2 norm
    x_sq = x * x
    x_sq_sum = tl.sum(x_sq, axis=0)
    x_norm = tl.sqrt(x_sq_sum)
    x_norm = tl.max(x_norm, 1e-8)  # Avoid division by zero

    # Normalize
    out = x / x_norm
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        B, N, D = x.shape

        # Flatten x to BN x D
        x = x.view(-1, D)  # B*N x D

        # Ensure we're on GPU
        x = x.contiguous()
        if x.device != self.clusters.device:
            raise ValueError(f"x.device {x.device} != cluster.device {self.clusters.device}")

        # Compute assignment: (BN x D) @ (D x (K+G)) -> BN x (K+G)
        # Use Triton matmul with fused bias (batch norm) and softmax
        assignment = torch.empty(x.shape[0], self.cluster_size + self.ghost_clusters, device=x.device, dtype=x.dtype)
        # We'll handle the batch norm and softmax later

        # Perform matmul with Triton
        M, K = x.shape[0], x.shape[1]
        K2, N2 = self.clusters.shape
        assert K == K2
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N2, meta['BLOCK_SIZE_N']),)
        matmul_kernel[grid](
            x, self.clusters, assignment,
            M, N2, K,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32
        )

        # Apply batch norm with Triton
        assignment = assignment.view(-1, self.cluster_size + self.ghost_clusters)
        assignment = self.batch_norm(assignment)
        assignment = assignment.view(-1, max_sample, self.cluster_size + self.ghost_clusters)

        # Apply softmax with Triton
        assignment = assignment.contiguous()
        assignment = assignment.view(-1, self.cluster_size + self.ghost_clusters)
        softmax_output = torch.empty_like(assignment)
        grid_softmax = lambda meta: (triton.cdiv(assignment.shape[0], meta['BLOCK_SIZE']),)
        softmax_kernel[grid_softmax](
            assignment, softmax_output,
            assignment.numel(), assignment.shape[1],
            BLOCK_SIZE=128
        )
        assignment = softmax_output.view(-1, max_sample, self.cluster_size)

        # Remove ghost assignments
        # assignment is now B x N x K
        a_sum = torch.sum(assignment, dim=1, keepdim=True)  # B x 1 x K

        # Compute a = a_sum * c_k (visual words)
        # This is a tensor contraction: B x 1 x K and 1 x D x K -> B x D x K
        # But we can fuse this with the next matmul
        a = a_sum * self.clusters2  # B x 1 x K x D -> B x D x K (not correct)

        # We need to transpose assignments to B x K x N
        assignment = assignment.transpose(1, 2)  # B x K x N

        # Reshape x to B x N x D
        x = x.view(B, N, D)  # BN x D -> B x N x D

        # Compute vlad = assignment @ x: B x K x N @ B x N x D -> B x K x D
        vlad = torch.empty(B, self.cluster_size, D, device=x.device, dtype=x.dtype)
        # Use Triton matmul again
        grid_vlad = lambda meta: (triton.cdiv(B * self.cluster_size, meta['BLOCK_SIZE_M']) * triton.cdiv(D, meta['BLOCK_SIZE_N']),)
        matmul_kernel[grid_vlad](
            assignment, x, vlad,
            B * self.cluster_size, D, N,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32
        )

        # vlad is B x K x D, transpose to B x D x K
        vlad = vlad.transpose(1, 2)  # B x D x K

        # Subtract a (visual words) from vlad
        # vlad = vlad - a
        # a is B x D x K, vlad is B x D x K
        vlad = vlad - a

        # L2 norm with Triton
        vlad = vlad.contiguous()
        vlad = vlad.view(-1, self.cluster_size * self.feature_size)
        vlad_norm = torch.empty_like(vlad)
        grid_norm = lambda meta: (triton.cdiv(vlad.numel(), meta['BLOCK_SIZE']),)
        l2_normalize_kernel[grid_norm](
            vlad, vlad_norm,
            vlad.numel(),
            BLOCK_SIZE=128
        )
        vlad = vlad_norm

        # Final flatten and normalize
        # Already flattened and normalized above
        return vlad