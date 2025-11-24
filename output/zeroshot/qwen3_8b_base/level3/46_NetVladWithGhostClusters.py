import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_softmax_kernel(
    x_ptr,  # Pointer to input tensor (B x N x D)
    clusters_ptr,  # Pointer to clusters (D x K)
    out_ptr,  # Pointer to output tensor (B x K x D)
    batch_size,  # B
    num_features,  # N
    feature_size,  # D
    cluster_size,  # K
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the block offset
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < (batch_size * num_features * feature_size)
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute assignment
    assignment = tl.dot(x, clusters_ptr)
    # Apply softmax
    assignment = F.softmax(assignment, dim=1)
    # Apply mask to remove ghost clusters
    assignment = assignment[:, :cluster_size]
    # Reshape to B x N x K
    assignment = assignment.reshape(batch_size, num_features, cluster_size)
    # Compute a_sum
    a_sum = tl.sum(assignment, axis=1, keepdim=True)
    # Compute a
    a = a_sum * clusters_ptr
    # Compute vlad
    vlad = tl.dot(assignment.transpose(1, 2), x)
    vlad = vlad.transpose(1, 2)
    vlad = vlad - a
    # Apply L2 normalization
    vlad = F.normalize(vlad)
    # Reshape to B x DK
    vlad = vlad.reshape(batch_size, cluster_size * feature_size)
    vlad = F.normalize(vlad)
    # Store the result
    tl.store(out_ptr + offsets, vlad, mask=mask)


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
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

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
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Use Triton kernel for matmul and softmax
        assignment = triton_matmul_softmax(x, self.clusters, self.cluster_size, self.feature_size)
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = torch.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = torch.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK