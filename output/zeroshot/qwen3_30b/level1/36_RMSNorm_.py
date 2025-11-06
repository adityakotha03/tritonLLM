import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    x_stride,
    y_stride,
    batch_size,
    num_features,
    dim1,
    dim2,
    eps,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Calculate the total number of elements in the feature dimension
    total_elements = num_features * dim1 * dim2
    # Each block handles BLOCK_SIZE elements
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Calculate the total number of elements in the batch
    total_batch_elements = batch_size * total_elements

    # Compute the current position in the flattened tensor
    # Each element is mapped to a (batch, feature, dim1, dim2) coordinate
    idx = offsets % total_elements
    feature_idx = idx // (dim1 * dim2)
    dim1_idx = (idx // dim2) % dim1
    dim2_idx = idx % dim2

    # Compute the batch index
    batch_idx = offsets // total_elements

    # Ensure we are within bounds
    valid = offsets < total_batch_elements
    mask = valid

    # Calculate the global indices for loading
    x_offset = batch_idx * x_stride + feature_idx * (dim1 * dim2) + dim1_idx * dim2 + dim2_idx
    x_ptr_offsets = x_ptr + x_offset

    # Load the input value
    x = tl.load(x_ptr_offsets, mask=mask, other=0.0)

    # Compute the mean of squares along the feature dimension (using shared memory for partial sums)
    # Use shared memory to reduce global memory traffic
    pid = tl.program_id(1)  # tile_id
    tile_offset = pid * TILE_SIZE
    tile_mask = tl.arange(0, TILE_SIZE) < (num_features - tile_offset)
    tile_indices = tile_offset + tl.arange(0, TILE_SIZE)
    tile_mask = tile_mask & (tile_indices < num_features)

    # Use shared memory for partial sum of squares
    shared_sum = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    for i in range(0, num_features, TILE_SIZE):
        tile_id = i // TILE_SIZE
        tile_idx = tile_id * TILE_SIZE + tl.arange(0, TILE_SIZE)
        tile_mask = tile_idx < num_features

        # Compute local square and sum
        local_idx = tile_idx
        x_local_offset = batch_idx * x_stride + local_idx * (dim1 * dim2) + dim1_idx * dim2 + dim2_idx
        x_local = tl.load(x_ptr + x_local_offset, mask=tile_mask & valid, other=0.0)
        x_local_sq = x_local * x_local
        tl.store(shared_sum + tile_idx % TILE_SIZE, x_local_sq, mask=tile_mask)

    # Use reduction to compute the sum across feature dimension
    sum_sq = tl.sum(shared_sum, axis=0)

    # Reduce sum across all tiles (we assume a single tile for simplicity)
    # Use block-wide reduction with shared memory
    # For now, we assume TILE_SIZE is small and fits in shared memory
    # Use warp shuffle for final reduction
    sum_sq = tl.broadcast(sum_sq, (1,))

    # Reduce sum across all features (using reduce across warps)
    sum_sq = tl.reduce(sum_sq, axis=0)

    # Compute the mean
    mean_sq = sum_sq / num_features

    # Compute RMS with epsilon
    rms = tl.sqrt(mean_sq + eps)

    # Normalize: x / rms
    y = x / rms

    # Store the output
    y_offset = batch_idx * y_stride + feature_idx * (dim1 * dim2) + dim1_idx * dim2 + dim2_idx
    y_ptr_offsets = y_ptr + y_offset
    tl.store(y_ptr_offsets, y, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS normalization using a Triton kernel with operator fusion and shared memory optimization.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    batch_size, num_features, dim1, dim2 = x.shape

    # Total elements in feature + spatial dimensions
    total_elements = num_features * dim1 * dim2
    total_batch_elements = batch_size * total_elements

    # Output tensor
    y = torch.empty_like(x)

    # Configure kernel grid and block size
    BLOCK_SIZE = 128  # Recommended block size for A100 (powers of 2)
    TILE_SIZE = 32   # Must divide num_features; use small tiles for shared memory

    # Compute number of blocks
    num_blocks = (total_batch_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Define grid function
    grid = lambda meta: (num_blocks, (meta["num_features"] + meta["TILE_SIZE"] - 1) // meta["TILE_SIZE"])

    # Launch kernel
    rms_norm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        x_stride=x.stride(0),
        y_stride=y.stride(0),
        batch_size=batch_size,
        num_features=num_features,
        dim1=dim1,
        dim2=dim2,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE
    )

    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'TILE_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'TILE_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'TILE_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'TILE_SIZE': 16}, num_stages=3, num_warps=4),
    ],
    key=['num_features', 'dim1', 'dim2'],
)
def triton_rms_norm_autotuned(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Autotuned version of RMS normalization using Triton kernels.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    batch_size, num_features, dim1, dim2 = x.shape

    total_elements = num_features * dim1 * dim2
    total_batch_elements = batch_size * total_elements

    y = torch.empty_like(x)

    # Use autotuned BLOCK_SIZE and TILE_SIZE
    # We will fix TILE_SIZE to 32 for stability and performance
    # BLOCK_SIZE will be autotuned
    grid = lambda meta: (
        (total_batch_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (meta['num_features'] + meta['TILE_SIZE'] - 1) // meta['TILE_SIZE']
    )

    # Launch the autotuned kernel
    rms_norm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        x_stride=x.stride(0),
        y_stride=y.stride(0),
        batch_size=batch_size,
        num_features=num_features,
        dim1=dim1,
        dim2=dim2,
        eps=eps,
        BLOCK_SIZE=128,  # Placeholder, will be replaced by autotuner
        TILE_SIZE=32
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the autotuned Triton kernel for optimal performance
        return triton_rms_norm_autotuned(x, self.eps)