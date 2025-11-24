import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batch_norm2d_kernel(
    x_ptr,        # Input tensor (batch, C, H, W)
    gamma_ptr,    # Gamma parameter (C,)
    beta_ptr,     # Beta parameter (C,)
    mean_ptr,     # Mean (C,)
    var_ptr,      # Variance (C,)
    out_ptr,      # Output tensor (batch, C, H, W)
    batch_size: tl.constexpr,
    num_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of spatial elements
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Ensure we don't exceed the number of channels
    if channel_idx >= num_channels:
        return
    
    # Compute the spatial indices
    row_start = batch_idx * height
    row_end = row_start + height
    col_start = 0
    col_end = width
    
    # Create a range of spatial indices
    row_ids = tl.arange(0, height)
    col_ids = tl.arange(0, width)
    
    # Broadcast over spatial dimensions
    row_offsets = row_ids[:, None] * width + col_ids[None, :]
    row_offsets = row_offsets.to(tl.int32)
    
    # Load input data for this channel
    x = tl.load(x_ptr + batch_idx * num_channels * height * width + channel_idx * height * width + row_offsets, 
                mask=row_offsets < height * width, other=0.0)
    
    # Compute the mean and variance (in a fused way for efficiency)
    # We assume mean and var are already computed and passed in
    mean_val = tl.load(mean_ptr + channel_idx, mask=channel_idx < num_channels, other=0.0)
    var_val = tl.load(var_ptr + channel_idx, mask=channel_idx < num_channels, other=1.0)
    
    # Compute normalization
    inv_std = 1.0 / tl.sqrt(var_val + 1e-5)
    gamma_val = tl.load(gamma_ptr + channel_idx, mask=channel_idx < num_channels, other=1.0)
    beta_val = tl.load(beta_ptr + channel_idx, mask=channel_idx < num_channels, other=0.0)
    
    # Apply normalization: (x - mean) / sqrt(var) * gamma + beta
    norm_x = (x - mean_val) * inv_std
    out = norm_x * gamma_val + beta_val
    
    # Store result
    tl.store(out_ptr + batch_idx * num_channels * height * width + channel_idx * height * width + row_offsets, out, mask=row_offsets < height * width)


@triton.jit
def conv2d_kernel(
    x_ptr,        # Input tensor (batch, C_in, H, W)
    w_ptr,        # Weight tensor (C_out, C_in, 1, 1)
    out_ptr,      # Output tensor (batch, C_out, H, W)
    batch_size: tl.constexpr,
    num_input_channels: tl.constexpr,
    num_output_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output channels
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    if out_channel_idx >= num_output_channels:
        return
    
    # Load weight for this output channel
    w = tl.load(w_ptr + out_channel_idx * num_input_channels + tl.arange(0, num_input_channels), 
                mask=tl.arange(0, num_input_channels) < num_input_channels, other=0.0)
    
    # Spatial indices
    row_ids = tl.arange(0, height)
    col_ids = tl.arange(0, width)
    
    # Compute spatial offsets
    row_offsets = row_ids[:, None] * width + col_ids[None, :]
    row_offsets = row_offsets.to(tl.int32)
    
    # Load input data for all input channels
    input_data = tl.load(x_ptr + batch_idx * num_input_channels * height * width + 
                         tl.arange(0, num_input_channels)[:, None] * height * width + row_offsets,
                         mask=row_offsets < height * width, other=0.0)
    
    # Compute output via dot product with weights
    # For 1x1 conv, it's just a linear transformation
    out = tl.zeros((height, width), dtype=tl.float32)
    for i in range(num_input_channels):
        out += input_data[i] * w[i]
    
    # Store output
    tl.store(out_ptr + batch_idx * num_output_channels * height * width + out_channel_idx * height * width + row_offsets, out, mask=row_offsets < height * width)


@triton.jit
def avg_pool2d_kernel(
    x_ptr,        # Input tensor (batch, C, H, W)
    out_ptr,      # Output tensor (batch, C, H//2, W//2)
    batch_size: tl.constexpr,
    num_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of spatial elements
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    if channel_idx >= num_channels:
        return
    
    # Compute output spatial indices
    row_ids = tl.arange(0, height // 2)
    col_ids = tl.arange(0, width // 2)
    
    # Compute spatial offsets in output
    out_row_offsets = row_ids[:, None] * (width // 2) + col_ids[None, :]
    
    # Compute input spatial offsets (each output pixel corresponds to 2x2 block)
    row_offsets = row_ids[:, None] * 2 * width + col_ids[None, :]
    col_offsets = col_ids[None, :] * 2 * width + row_ids[:, None]
    
    # Load input data for each 2x2 block
    input_data = tl.load(x_ptr + batch_idx * num_channels * height * width + 
                         channel_idx * height * width + row_offsets, 
                         mask=row_offsets < height * width, other=0.0)
    
    # Compute average over 2x2 block
    block_sum = tl.sum(input_data, axis=(0, 1))
    block_count = 4
    avg_val = block_sum / block_count
    
    # Store result
    tl.store(out_ptr + batch_idx * num_channels * (height // 2) * (width // 2) + channel_idx * (height // 2) * (width // 2) + out_row_offsets, avg_val, mask=out_row_offsets < (height // 2) * (width // 2))


def triton_batch_norm2d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
    """
    Custom batch norm 2D kernel using Triton.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and mean.is_cuda and var.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    batch_size, num_channels, height, width = x.shape
    BLOCK_SIZE = 128

    # Grid: (batch, channel) blocks
    grid = lambda meta: ((batch_size, num_channels))

    batch_norm2d_kernel[grid](
        x_ptr=x.data_ptr(),
        gamma_ptr=gamma.data_ptr(),
        beta_ptr=beta.data_ptr(),
        mean_ptr=mean.data_ptr(),
        var_ptr=var.data_ptr(),
        out_ptr=torch.empty_like(x).data_ptr(),
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x


def triton_conv2d(x: torch.Tensor, w: torch.Tensor):
    """
    Custom 1x1 Conv2D kernel using Triton.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, num_input_channels, height, width = x.shape
    num_output_channels = w.shape[0]
    BLOCK_SIZE = 128

    # Grid: (batch, output_channel)
    grid = lambda meta: ((batch_size, num_output_channels))

    conv2d_kernel[grid](
        x_ptr=x.data_ptr(),
        w_ptr=w.data_ptr(),
        out_ptr=torch.empty(batch_size, num_output_channels, height, width, dtype=x.dtype).data_ptr(),
        batch_size=batch_size,
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x


def triton_avg_pool2d(x: torch.Tensor):
    """
    Custom average pooling kernel using Triton.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    batch_size, num_channels, height, width = x.shape
    BLOCK_SIZE = 128

    # Grid: (batch, channel)
    grid = lambda meta: ((batch_size, num_channels))

    avg_pool2d_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=torch.empty(batch_size, num_channels, height // 2, width // 2, dtype=x.dtype).data_ptr(),
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

        # Pre-allocate parameters for batch norm and conv
        self.gamma = nn.Parameter(torch.ones(num_input_features))
        self.beta = nn.Parameter(torch.zeros(num_input_features))
        self.mean = nn.Parameter(torch.zeros(num_input_features))
        self.var = nn.Parameter(torch.ones(num_input_features))

        # 1x1 Conv weight
        self.weight = nn.Parameter(torch.randn(num_output_features, num_input_features, 1, 1))

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        # 1. BatchNorm2d
        x = triton_batch_norm2d(x, self.gamma, self.beta, self.mean, self.var)

        # 2. ReLU activation (we keep this as PyTorch for now, since it's not memory-bound)
        x = F.relu(x, inplace=True)

        # 3. 1x1 Conv
        x = triton_conv2d(x, self.weight)

        # 4. AvgPool2d
        x = triton_avg_pool2d(x)

        return x