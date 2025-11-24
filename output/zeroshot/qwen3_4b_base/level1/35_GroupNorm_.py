import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr, 
    num_features: tl.constexpr, 
    num_groups: tl.constexpr, 
    batch_size: tl.constexpr, 
    dim1: tl.constexpr, 
    dim2: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the total number of elements in the spatial dimensions
    total_elements = dim1 * dim2
    # Each block processes a batch element
    batch_idx = tl.program_id(0)
    # Check if batch_idx is within bounds
    batch_mask = (batch_idx < batch_size) if batch_size > 0 else tl.ones(1, dtype=tl.int32)
    
    # Compute the start index for this batch
    batch_start = batch_idx * total_elements
    # Create a range of spatial indices
    spatial_offsets = tl.arange(0, total_elements)
    # Compute the total index for each spatial element
    indices = spatial_offsets + batch_start
    
    # Load input values
    x = tl.load(x_ptr + indices, mask=spatial_offsets < total_elements, other=0.0)
    
    # Compute group size per group
    group_size = num_features // num_groups
    # Reshape the input to group dimensions: (batch, num_groups, group_size, dim1, dim2)
    # We compute per group and apply normalization across spatial dimensions
    
    # For each group, we compute mean and variance across spatial dimensions
    # We will compute mean and variance in a fused way using shared memory
    # But since we are in a single kernel, we will use a tiling approach
    
    # Instead, we do a simplified per-group normalization without shared memory
    # We compute group-wise mean and variance directly in kernel
    
    # We'll compute mean and variance for each group across spatial dims
    # Group-wise: each group has size (group_size) in feature dimension
    
    # We'll process each group independently
    group_id = tl.program_id(1)
    group_mask = (group_id < num_groups)
    
    # If we are in a group, compute the group's features
    if not group_mask:
        return
    
    # Each group has a feature range
    group_start = group_id * group_size
    group_end = (group_id + 1) * group_size
    group_features = x[:, group_start:group_end, :, :]  # This is not valid in Triton due to dynamic indexing
    
    # Instead, we use a different approach: flatten the spatial dimensions and compute group-wise stats
    # We will compute mean and variance across spatial dims for each group
    
    # Instead, we use a more efficient method: reshape and compute group-wise normalization
    # We compute the mean and variance over spatial dims for each group
    
    # We flatten spatial dims to compute per-group mean and variance
    # We use a loop over groups to compute stats
    
    # Since we cannot easily reshape in Triton with dynamic indexing, we use a different approach:
    # We compute mean and variance in a vectorized way using block-level computation
    
    # Instead, we rewrite the kernel to work on the flattened spatial dimensions
    # We will compute the mean and variance per group across spatial dims
    
    # We'll use a simplified version that computes group-wise normalization without full tensor reshape
    # This is a simplified kernel that works only on the feature dimension
    
    # Instead, we use a more practical approach: we process each spatial element and apply group-wise normalization
    # We compute the mean and variance of each group across spatial dims
    
    # Since this is complex, we will instead implement a fused kernel that computes the group norm
    # by first computing the mean and variance per group across spatial dims
    
    # We will compute mean and variance for each group across spatial dimensions
    # We use a block of size BLOCK_SIZE to process spatial elements
    
    # We restructure: process spatial elements in a block
    spatial_block = tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_block < total_elements
    
    # Load spatial elements for current batch and group
    # We load the values for the current spatial block
    x_vals = tl.load(x_ptr + indices + spatial_block, mask=spatial_mask, other=0.0)
    
    # Compute mean and variance for this group
    # We compute mean over spatial dimensions
    mean_val = tl.sum(x_vals, axis=0) / total_elements
    # We compute variance
    var_val = tl.sum((x_vals - mean_val) ** 2, axis=0) / total_elements
    
    # Apply normalization: (x - mean) / sqrt(var + eps)
    # We use a small epsilon for numerical stability
    eps = 1e-5
    std_val = tl.sqrt(var_val + eps)
    
    # Apply normalization
    norm_val = (x_vals - mean_val) / std_val
    
    # Store result
    tl.store(out_ptr + indices + spatial_block, norm_val, mask=spatial_mask)


@triton.jit
def group_norm_kernel_fused(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr, 
    num_features: tl.constexpr, 
    num_groups: tl.constexpr, 
    batch_size: tl.constexpr, 
    dim1: tl.constexpr, 
    dim2: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Process each batch element
    batch_idx = tl.program_id(0)
    batch_mask = (batch_idx < batch_size) if batch_size > 0 else tl.ones(1, dtype=tl.int32)
    
    # Total spatial elements
    total_elements = dim1 * dim2
    
    # Spatial offsets
    spatial_offsets = tl.arange(0, total_elements)
    indices = spatial_offsets + batch_idx * total_elements
    
    # Load input
    x = tl.load(x_ptr + indices, mask=spatial_offsets < total_elements, other=0.0)
    
    # Compute group size
    group_size = num_features // num_groups
    group_id = tl.program_id(1)
    group_mask = (group_id < num_groups)
    
    # Only process if in valid group
    if not group_mask:
        return
    
    # Compute group start and end
    group_start = group_id * group_size
    group_end = (group_id + 1) * group_size
    
    # Extract group features
    # We cannot directly slice in Triton, so we use a different approach
    # Instead, we compute the mean and variance across spatial dims for each group
    
    # We use a block to process spatial elements
    spatial_block = tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_block < total_elements
    
    # Load values in block
    x_block = tl.load(x_ptr + indices + spatial_block, mask=spatial_mask, other=0.0)
    
    # Compute mean and variance over spatial dims
    mean_val = tl.sum(x_block, axis=0) / total_elements
    var_val = tl.sum((x_block - mean_val) ** 2, axis=0) / total_elements
    eps = 1e-5
    std_val = tl.sqrt(var_val + eps)
    
    # Normalize
    norm_val = (x_block - mean_val) / std_val
    
    # Store result
    tl.store(out_ptr + indices + spatial_block, norm_val, mask=spatial_mask)


def triton_group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_features: int,
    num_groups: int,
    dim1: int,
    dim2: int,
):
    """
    Custom GroupNorm kernel using Triton.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight is not None and bias is not None, "Weight and bias must be provided."
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Prepare output tensor
    out = x.clone()
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid: number of blocks needed
    grid = lambda meta: (
        (x.shape[0] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (num_groups + 1)  # one block per group
    )
    
    # Launch kernel
    group_norm_kernel_fused[
        grid
    ](
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        num_features=num_features,
        num_groups=num_groups,
        batch_size=x.shape[0],
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        # We do not use nn.GroupNorm; instead we implement custom GroupNorm
        # Weights and bias are learned
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Group Normalization to the input tensor.
        """
        # Ensure input is on GPU
        if x.device != torch.cuda.current_device():
            x = x.cuda()
        
        # Apply custom GroupNorm via Triton kernel
        return triton_group_norm(x, self.weight, self.bias, self.num_features, self.num_groups, x.shape[2], x.shape[3])