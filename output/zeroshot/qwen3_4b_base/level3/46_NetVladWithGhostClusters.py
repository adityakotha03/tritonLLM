import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    x_ptr,  # (B x N x D) -> (BN x D)
    clusters_ptr,  # (D x (K+G))
    out_ptr,  # (BN x (K+G))
    n_samples,  # N
    d,  # D
    k_plus_g,  # K + G
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples * d
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    clusters = tl.load(clusters_ptr + offsets, mask=mask, other=0.0)
    out = tl.dot(x, clusters)  # (BN x D) @ (D x (K+G)) -> (BN x (K+G))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def softmax_kernel(
    x_ptr,  # (BN x (K+G))
    out_ptr,  # (BN x (K+G))
    n_samples,  # N
    k_plus_g,  # K + G
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples * k_plus_g
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    exp_x = tl.exp(x - tl.max(x, 0.0))  # avoid overflow
    sum_exp = tl.sum(exp_x, axis=1, keep_dim=True)
    softmax = exp_x / sum_exp
    tl.store(out_ptr + offsets, softmax, mask=mask)


@triton.jit
def vlad_kernel(
    assignment_ptr,  # (B x N x K)
    x_ptr,  # (B x N x D)
    a_ptr,  # (B x K x D)
    vlad_ptr,  # (B x K x D)
    n_samples,  # N
    k,  # K
    d,  # D
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples * k
    assignment = tl.load(assignment_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    vlad = tl.dot(assignment, x) - a
    tl.store(vlad_ptr + offsets, vlad, mask=mask)


@triton.jit
def normalize_kernel(
    x_ptr,  # (B x D x K)
    out_ptr,  # (B x DK)
    b,  # B
    d,  # D
    k,  # K
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < b * d * k
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    norm = tl.sqrt(tl.sum(x * x, axis=1, keep_dim=True))
    x = x / norm
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_matmul(x: torch.Tensor, clusters: torch.Tensor):
    assert x.is_cuda and clusters.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    clusters = clusters.contiguous()

    b, n, d = x.shape
    k_plus_g = clusters.shape[1]
    n_elements = b * n * d

    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty((b * n, k_plus_g), device=x.device, dtype=clusters.dtype)
    matmul_kernel[grid](x, clusters, out, n, d, k_plus_g, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    b, n, k_plus_g = x.shape
    n_elements = b * n * k_plus_g

    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty_like(x)
    softmax_kernel[grid](x, out, n, k_plus_g, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_vlad(assignment: torch.Tensor, x: torch.Tensor, a: torch.Tensor):
    assert assignment.is_cuda and x.is_cuda and a.is_cuda, "All tensors must be on CUDA."
    assignment = assignment.contiguous()
    x = x.contiguous()
    a = a.contiguous()

    b, n, k = assignment.shape
    d = x.shape[2]
    n_elements = b * n * k

    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    vlad = torch.empty((b, k, d), device=x.device, dtype=x.dtype)
    vlad_kernel[grid](assignment, x, a, vlad, n, k, d, BLOCK_SIZE=BLOCK_SIZE)
    return vlad


def triton_normalize(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    b, d, k = x.shape
    n_elements = b * d * k

    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty((b, d * k), device=x.device, dtype=x.dtype)
    normalize_kernel[grid](x, out, b, d, k, BLOCK_SIZE=BLOCK_SIZE)
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
        """Aggregates feature maps into a fixed size representation. In the following
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
        assignment = triton_matmul(x, self.clusters)  # (BN x (K+G))
        assignment = self.batch_norm(assignment)  # (BN x (K+G))

        # Step 2: Apply softmax
        assignment = triton_softmax(assignment)  # (BN x (K+G))
        assignment = assignment[:, :self.cluster_size]  # Remove ghost assignments
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # B x N x K

        # Step 3: Compute a_sum and a
        a_sum = assignment.sum(dim=1, keepdim=True)  # B x 1 x K
        a = a_sum * self.clusters2  # B x 1 x D -> B x K x D

        # Step 4: Compute vlad
        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N
        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = triton_vlad(assignment, x, a)  # (B x K x D)
        vlad = vlad.transpose(1, 2)  # B x D x K

        # Step 5: Normalize vlad
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # B x DK
        vlad = triton_normalize(vlad)  # B x DK
        return vlad  # B x DK