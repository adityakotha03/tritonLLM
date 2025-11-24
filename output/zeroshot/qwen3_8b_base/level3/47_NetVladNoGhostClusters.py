import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_softmax_kernel(
    x_ptr,  # Pointer to input tensor (B x N x D)
    clusters_ptr,  # Pointer to clusters (D x K)
    out_ptr,  # Pointer to output (B x K x D)
    batch_size,  # B
    num_features,  # N
    feature_size,  # D
    cluster_size,  # K
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the index in the flattened input tensor (B x N x D -> BN x D)
    idx = (offsets // feature_size) * feature_size + (offsets % feature_size)
    idx = idx + tl.arange(0, BLOCK_SIZE) * feature_size
    idx = idx + tl.arange(0, BLOCK_SIZE) * feature_size * num_features
    idx = idx + tl.arange(0, BLOCK_SIZE) * feature_size * num_features * batch_size
    idx = idx + tl.arange(0, BLOCK_SIZE) * feature_size * num_features * batch_size

    # Load input data
    x = tl.load(x_ptr + idx, mask=offsets < batch_size * num_features * feature_size, other=0.0)

    # Compute assignment: x @ clusters
    assignment = tl.dot(x, clusters_ptr, axis=0)
    assignment = F.softmax(assignment, dim=0)

    # Apply mask for ghost clusters
    assignment = assignment[:, :cluster_size]

    # Compute a_sum = sum(assignment, dim=1)
    a_sum = tl.sum(assignment, axis=1)

    # Compute a = a_sum * clusters2
    a = a_sum * clusters_ptr

    # Compute vlad = assignment @ x
    vlad = tl.dot(assignment, x, axis=1)

    # Compute vlad = vlad - a
    vlad = vlad - a

    # Normalize vlad
    vlad = F.normalize(vlad)

    # Store output
    out_idx = (pid * feature_size) + tl.arange(0, BLOCK_SIZE)
    out_idx = out_idx + tl.arange(0, BLOCK_SIZE) * feature_size * cluster_size
    out_idx = out_idx + tl.arange(0, BLOCK_SIZE) * feature_size * cluster_size * batch_size
    tl.store(out_ptr + out_idx, vlad, mask=offsets < batch_size * num_features * feature_size)


def triton_matmul_softmax(x: torch.Tensor, clusters: torch.Tensor, cluster_size: int, feature_size: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and clusters.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    clusters = clusters.contiguous()

    # Prepare output tensor
    out = torch.empty((x.size(0), cluster_size, feature_size), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_softmax_kernel[grid](x, clusters, out, x.size(0), x.size(1), x.size(2), cluster_size, BLOCK_SIZE=BLOCK_SIZE)
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
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))

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
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Compute assignment and softmax using Triton kernel
        assignment = self.batch_norm(self.triton_matmul_softmax(x, self.clusters))
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = torch.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = self.triton_matmul_softmax(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK

    def triton_matmul_softmax(self, x: torch.Tensor, clusters: torch.Tensor):
        return triton_matmul_softmax(x, clusters, self.cluster_size, self.feature_size)