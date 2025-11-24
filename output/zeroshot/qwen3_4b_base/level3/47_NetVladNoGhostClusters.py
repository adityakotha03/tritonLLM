import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    x_ptr,  # (BN x D) -> B x N x D
    w_ptr,  # (D x K) -> D x (K+G)
    out_ptr,  # (BN x K) -> B x N x K
    n_samples,  # N
    n_features,  # D
    n_clusters,  # K+G
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of N samples
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples
    # Load x: (N x D) -> each thread loads one row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load weights: (D x K) -> each thread loads one column
    w = tl.load(w_ptr + tl.arange(0, n_clusters) * n_features, mask=tl.arange(0, n_clusters) < n_clusters, other=0.0)
    # Compute dot product: (N x D) @ (D x K) -> (N x K)
    out = tl.zeros((BLOCK_SIZE, n_clusters), dtype=tl.float16)
    for i in range(0, n_features, BLOCK_SIZE):
        # Load a slice of D features
        x_slice = x
        w_slice = w
        # Compute dot product
        out = out + tl.dot(x_slice, w_slice)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def softmax_kernel(
    x_ptr,  # (BN x K)
    out_ptr,  # (BN x K)
    n_samples,  # N
    n_clusters,  # K
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute softmax in log space to avoid overflow
    max_val = tl.max(x, axis=1, keepdim=True)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=1, keepdim=True)
    softmax = exp_x / sum_exp
    # Store result
    tl.store(out_ptr + offsets, softmax, mask=mask)


@triton.jit
def vlad_kernel(
    assignment_ptr,  # (B x N x K)
    x_ptr,  # (B x N x D)
    a_ptr,  # (B x K x D)
    vlad_ptr,  # (B x K x D)
    n_samples,  # N
    n_clusters,  # K
    feature_size,  # D
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples
    # Load assignment
    assignment = tl.load(assignment_ptr + offsets, mask=mask, other=0.0)
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load a
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    # Compute vlad: assignment @ x - a
    vlad = tl.zeros((BLOCK_SIZE, n_clusters, feature_size), dtype=tl.float16)
    for i in range(0, n_clusters, BLOCK_SIZE):
        # Load assignment slice
        assign_slice = assignment
        x_slice = x
        a_slice = a
        # Compute dot product
        vlad = vlad + tl.dot(assign_slice, x_slice) - a_slice
    # Store result
    tl.store(vlad_ptr + offsets, vlad, mask=mask)


@triton.jit
def normalize_kernel(
    x_ptr,  # (B x K x D)
    out_ptr,  # (B x K x D)
    n_batch,  # B
    n_clusters,  # K
    feature_size,  # D
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_batch
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute L2 norm per cluster
    norm_sq = tl.sum(x * x, axis=2, keepdim=True)
    norm = tl.sqrt(norm_sq)
    # Normalize
    x_norm = x / norm
    # Store result
    tl.store(out_ptr + offsets, x_norm, mask=mask)


def triton_matmul(x: torch.Tensor, w: torch.Tensor):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    n_samples = x.size(1)
    n_features = x.size(2)
    n_clusters = w.size(1)

    # Output shape: (B x N x K)
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = lambda meta: ((n_samples + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    matmul_kernel[grid](x, w, out, n_samples, n_features, n_clusters, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda and x.device == torch.device("cuda"), "Tensors must be on CUDA"
    x = x.contiguous()
    n_samples = x.size(1)
    n_clusters = x.size(2)

    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_samples + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softmax_kernel[grid](x, out, n_samples, n_clusters, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_vlad(assignment: torch.Tensor, x: torch.Tensor, a: torch.Tensor):
    assert assignment.is_cuda and x.is_cuda and a.is_cuda, "Tensors must be on CUDA"
    assignment = assignment.contiguous()
    x = x.contiguous()
    a = a.contiguous()

    n_samples = assignment.size(1)
    n_clusters = assignment.size(2)
    feature_size = x.size(2)

    vlad = torch.empty(assignment.size(0), n_clusters, feature_size, device=assignment.device, dtype=torch.float16)

    BLOCK_SIZE = 128
    grid = lambda meta: ((n_samples + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    vlad_kernel[grid](assignment, x, a, vlad, n_samples, n_clusters, feature_size, BLOCK_SIZE=BLOCK_SIZE)
    return vlad


def triton_normalize(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    n_batch = x.size(0)
    n_clusters = x.size(1)
    feature_size = x.size(2)

    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_batch + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    normalize_kernel[grid](x, out, n_batch, n_clusters, feature_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters).cuda())
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size).cuda())
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
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            raise ValueError(f"x.device {x.device} != cluster.device {self.clusters.device}")

        # Step 1: Compute assignment via matmul
        assignment = triton_matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        # Step 2: Apply softmax
        assignment = triton_softmax(assignment)  # BN x (K+G) -> BN x (K+G)
        assignment = assignment[:, :self.cluster_size]  # Remove ghost assignments
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K

        # Step 3: Compute sum over features for each cluster
        a_sum = assignment.sum(dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2  # B x 1 x K -> B x 1 x D

        # Step 4: Transpose assignment for vlad
        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        # Step 5: Reshape x
        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D

        # Step 6: Compute vlad
        vlad = triton_vlad(assignment, x, a)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a  # B x D x K - B x 1 x K -> B x D x K

        # Step 7: Normalize per cluster (L2)
        vlad = triton_normalize(vlad)  # B x D x K -> B x D x K

        # Step 8: Flatten and normalize
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)

        return vlad  # B x DK