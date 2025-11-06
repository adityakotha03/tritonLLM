import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    y_ptr,
    batch_size,
    num_features,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.float32,
):
    # Each block processes one feature channel across one batch element
    # Calculate which batch and feature this block is processing
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    
    # Check bounds
    if pid_batch >= batch_size or pid_feature >= num_features:
        return
    
    # Total elements per channel
    elements_per_channel = height * width
    
    # Calculate the start offset for this batch and feature
    offset = pid_batch * num_features * elements_per_channel + pid_feature * elements_per_channel
    
    # Create a range of offsets for this feature channel
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_channel
    
    # Load the data for this feature channel
    x = tl.load(x_ptr + offset + offsets, mask=mask, other=0.0)
    
    # Compute mean for this feature channel (reduce across H and W)
    # Use a block reduction across the H*W dimensions
    mean = tl.sum(x, axis=0) / elements_per_channel
    
    # Store mean for this feature and batch
    tl.store(mean_ptr + pid_batch * num_features + pid_feature, mean)
    
    # Compute variance
    # x - mean, squared
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / elements_per_channel
    
    # Store variance for this feature and batch
    tl.store(var_ptr + pid_batch * num_features + pid_feature, var)
    
    # Normalize: (x - mean) / sqrt(var + eps)
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = x_centered * inv_std
    
    # Store normalized output
    tl.store(y_ptr + offset + offsets, y, mask=mask)


@triton.jit
def instance_norm_affine_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    batch_size,
    num_features,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.float32,
):
    # Each block processes one feature channel across one batch element
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    
    if pid_batch >= batch_size or pid_feature >= num_features:
        return
    
    elements_per_channel = height * width
    offset = pid_batch * num_features * elements_per_channel + pid_feature * elements_per_channel
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_channel
    
    # Load data
    x = tl.load(x_ptr + offset + offsets, mask=mask, other=0.0)
    
    # Load mean and variance
    mean = tl.load(mean_ptr + pid_batch * num_features + pid_feature)
    var = tl.load(var_ptr + pid_batch * num_features + pid_feature)
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_centered = x - mean
    y = x_centered * inv_std
    
    # Apply affine transformation if weights and biases exist
    weight = tl.load(weight_ptr + pid_feature)
    bias = tl.load(bias_ptr + pid_feature)
    y = y * weight + bias
    
    # Store output
    tl.store(y_ptr + offset + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        # Affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA
        assert x.is_cuda, "Input tensor must be on CUDA"
        
        # Make sure x is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size, num_features, height, width = x.shape
        
        # Allocate intermediate buffers for mean and variance
        mean = torch.empty(batch_size, num_features, device=x.device, dtype=x.dtype)
        var = torch.empty(batch_size, num_features, device=x.device, dtype=x.dtype)
        
        # Set up Triton kernel configuration
        BLOCK_SIZE = 128
        eps = 1e-5
        
        # Grid for instance norm kernel
        grid = lambda meta: (batch_size, num_features)
        
        # Launch kernel to compute mean and variance
        instance_norm_kernel[grid](
            x_ptr=x,
            mean_ptr=mean,
            var_ptr=var,
            y_ptr=x,  # Temporarily use x as output buffer
            batch_size=batch_size,
            num_features=num_features,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
            eps=eps
        )
        
        # Launch affine kernel for final output
        out = torch.empty_like(x)
        instance_norm_affine_kernel[grid](
            x_ptr=x,
            mean_ptr=mean,
            var_ptr=var,
            y_ptr=out,
            weight_ptr=self.weight,
            bias_ptr=self.bias,
            batch_size=batch_size,
            num_features=num_features,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
            eps=eps
        )
        
        return out