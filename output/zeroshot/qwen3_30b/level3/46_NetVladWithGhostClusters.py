import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Each program processes a row of the input
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load the row
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Compute the maximum value for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Compute the exponential
    x = tl.exp(x)

    # Compute the sum
    x_sum = tl.sum(x, axis=0)

    # Compute the softmax
    y = x / x_sum

    # Store the output
    tl.store(y_ptr + row_idx * n_cols + col_offsets, y, mask=col_offsets < n_cols)


@triton.jit
def matmul_softmax_matmul_kernel(
    x_ptr,
    clusters_ptr,
    clusters2_ptr,
    a_ptr,
    vlad_ptr,
    batch_size,
    num_features,
    cluster_size,
    feature_size,
    ghost_clusters,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Shared memory for tiling
    shared_mem = tl.load(tl.memptr(tl.static_addressof(tl.static_alloc(16384, 16)), 0, 16384), cache=tl.cache_type.shared)

    # Indices
    pid = tl.program_id(0)
    block_idx = pid // (num_features + 1)
    block_row = block_idx % num_features
    block_col = block_idx // num_features
    thread_id = tl.program_id(1)

    # Tile size
    tile_size = TILE_SIZE

    # Initialize output
    accumulator = tl.zeros((tile_size, tile_size), dtype=tl.float32)

    # Compute indices
    start_idx = block_row * tile_size
    end_idx = min(start_idx + tile_size, num_features)
    num_elements = end_idx - start_idx

    # Get the row of x and the column of clusters
    x = tl.load(x_ptr + (start_idx * feature_size) + tl.arange(0, feature_size), mask=tl.arange(0, feature_size) < feature_size, other=0.0)
    clusters = tl.load(clusters_ptr + (tl.arange(0, cluster_size + ghost_clusters) * feature_size) + tl.arange(0, feature_size), mask=tl.arange(0, feature_size) < feature_size, other=0.0)
    
    # Matmul
    x = tl.dot(x, clusters, trans_b=True)
    
    # Apply softmax
    x = tl.log(x + 1e-8)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum
    
    # Remove ghost clusters
    x = x[:, :cluster_size]

    # Store assignment
    tl.store(a_ptr + (block_row * cluster_size) + tl.arange(0, cluster_size), x, mask=tl.arange(0, cluster_size) < cluster_size)

    # Compute the sum of assignments
    a_sum = tl.sum(x, axis=0)
    a_sum = tl.reshape(a_sum, (1, cluster_size))

    # Multiply with clusters2
    clusters2 = tl.load(clusters2_ptr + tl.arange(0, cluster_size * feature_size), mask=tl.arange(0, cluster_size * feature_size) < cluster_size * feature_size, other=0.0)
    a_sum = tl.reshape(a_sum, (1, cluster_size, 1))
    a_sum = tl.broadcast(a_sum, (1, cluster_size, feature_size))
    a_sum = tl.dot(a_sum, tl.reshape(clusters2, (cluster_size, feature_size)))

    # Transpose assignment and compute matmul
    x = tl.load(a_ptr + (block_row * cluster_size) + tl.arange(0, cluster_size), mask=tl.arange(0, cluster_size) < cluster_size, other=0.0)
    x = tl.reshape(x, (cluster_size, 1))
    x = tl.transpose(x, 0, 1)

    x = tl.dot(x, tl.load(x_ptr + (start_idx * feature_size) + tl.arange(0, feature_size), mask=tl.arange(0, feature_size) < feature_size, other=0.0))
    
    # Subtract a_sum
    vlad = x - a_sum

    # Store vlad
    tl.store(vlad_ptr + (block_row * cluster_size * feature_size) + tl.arange(0, cluster_size * feature_size), vlad, mask=tl.arange(0, cluster_size * feature_size) < cluster_size * feature_size)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters
        self.out_dim = cluster_size * feature_size

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))

    def forward(self, x, mask=None):
        # Flatten the input to B x N x D -> BN x D
        batch_size, num_features, feature_size = x.shape
        x = x.view(-1, self.feature_size)

        # Ensure contiguous tensors on GPU
        x = x.contiguous()
        clusters = self.clusters.contiguous()
        clusters2 = self.clusters2.contiguous()

        # Prepare output
        out = torch.empty(batch_size * num_features * self.cluster_size, device=x.device, dtype=x.dtype)

        # Use a larger BLOCK_SIZE for better occupancy
        BLOCK_SIZE = 1024
        TILE_SIZE = 32

        # Define grid for the kernel
        grid = lambda meta: (
            (num_features * (self.cluster_size + self.ghost_clusters + 1)) // meta["BLOCK_SIZE"] + 1,
            1,
        )

        # Launch the kernel
        matmul_softmax_matmul_kernel[grid](
            x_ptr=x,
            clusters_ptr=clusters,
            clusters2_ptr=clusters2,
            a_ptr=out,
            vlad_ptr=out,
            batch_size=batch_size,
            num_features=num_features,
            cluster_size=self.cluster_size,
            feature_size=feature_size,
            ghost_clusters=self.ghost_clusters,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE_SIZE=TILE_SIZE,
        )

        # Reshape to B x K x D and perform L2 normalization
        out = out.view(batch_size, num_features, self.cluster_size, self.feature_size)
        out = out.transpose(1, 2)  # B x D x K x N -> B x D x K
        out = F.normalize(out, p=2, dim=1)

        # Flatten and normalize again
        out = out.reshape(batch_size, self.cluster_size * self.feature_size)
        out = F.normalize(out, p=2, dim=1)

        return out