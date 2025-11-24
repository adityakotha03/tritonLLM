import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,           # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,          # pointer to weight tensor (out_channels, in_channels, 1, 1) or (out_channels, in_channels, 3, 3)
    bias_ptr,            # pointer to bias tensor (out_channels)
    output_ptr,          # pointer to output tensor (batch, out_channels, H, W)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Define the block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Define the spatial indices
    row_start = tl.program_id(2) * BLOCK_SIZE
    col_start = tl.program_id(3) * BLOCK_SIZE
    
    # Create the spatial indices
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to avoid out-of-bounds access
    row_mask = row_offsets < height
    col_mask = col_offsets < width
    
    # Compute valid spatial indices
    valid_row = row_offsets < height
    valid_col = col_offsets < width
    
    # Create mask for valid spatial positions
    mask = valid_row[:, None] & valid_col[:, None]
    
    # Load input features
    # Input: (batch, in_channels, H, W)
    # We process one spatial block at a time
    input_batch = batch_idx
    input_h = row_offsets
    input_w = col_offsets
    
    # For each output channel, compute the output
    # We use a 1x1 or 3x3 convolution
    if kernel_size == 1:
        # 1x1 convolution
        input_offset = input_batch * in_channels * height * width + \
                       tl.arange(0, in_channels)[:, None] * height * width + \
                       input_h[:, None] * width + input_w
        input_values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Weight: (out_channels, in_channels, 1, 1)
        weight_offset = out_channel_idx * in_channels + tl.arange(0, in_channels)
        weights = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Compute output
        output_val = tl.sum(input_values * weights, axis=0)
        
        # Add bias if exists
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_channel_idx, mask=mask, other=0.0)
            output_val += bias_val
        
    else:
        # 3x3 convolution with padding
        # Input: (batch, in_channels, H, W)
        # We need to handle padded input
        # For each spatial position, compute the convolution
        # We use a tiled approach with block size
        input_offset = input_batch * in_channels * height * width + \
                       tl.arange(0, in_channels)[:, None] * height * width + \
                       input_h[:, None] * width + input_w
        input_values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Weight: (out_channels, in_channels, 3, 3)
        weight_offset = out_channel_idx * in_channels * 9 + \
                        tl.arange(0, in_channels)[:, None] * 9 + \
                        (tl.arange(0, 3)[:, None] * 3 + tl.arange(0, 3))[:, :, None]
        weights = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Compute convolution
        # We use a 3x3 kernel
        # We loop over the kernel positions
        kernel_row = tl.arange(0, 3)
        kernel_col = tl.arange(0, 3)
        
        # Compute the output value
        output_val = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for i in range(3):
            for j in range(3):
                # Get the kernel position
                k_row = i
                k_col = j
                # Compute the input offset
                input_k_row = input_h + k_row - pad
                input_k_col = input_w + k_col - pad
                # Create mask for valid input
                k_row_mask = (input_k_row >= 0) & (input_k_row < height)
                k_col_mask = (input_k_col >= 0) & (input_k_col < width)
                k_mask = k_row_mask & k_col_mask
                # Load the input value
                input_k_offset = input_batch * in_channels * height * width + \
                                 tl.arange(0, in_channels)[:, None] * height * width + \
                                 input_k_row[:, None] * width + input_k_col
                input_k_val = tl.load(input_ptr + input_k_offset, mask=k_mask, other=0.0)
                # Load the weight
                weight_k_offset = out_channel_idx * in_channels * 9 + \
                                  tl.arange(0, in_channels)[:, None] * 9 + \
                                  (k_row[:, None] * 3 + k_col)[:, :, None]
                weight_k_val = tl.load(weight_ptr + weight_k_offset, mask=k_mask, other=0.0)
                # Accumulate
                output_val += input_k_val * weight_k_val
        
        # Add bias
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_channel_idx, mask=mask, other=0.0)
            output_val += bias_val
    
    # Store output
    output_offset = batch_idx * out_channels * height * width + out_channel_idx * height * width + row_offsets[:, None] * width + col_offsets
    tl.store(output_ptr + output_offset, output_val, mask=mask)


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


def triton_conv2d(
    input_tensor,
    weight_tensor,
    bias_tensor=None,
    kernel_size=1,
    pad=0,
    out_channels=None,
    in_channels=None,
    height=None,
    width=None,
    BLOCK_SIZE=128,
):
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight_tensor.is_cuda, "Weight tensor must be on CUDA"
    
    # Ensure contiguous memory
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    
    # Prepare output tensor
    if bias_tensor is not None:
        bias_tensor = bias_tensor.contiguous()
    
    batch_size = input_tensor.shape[0]
    in_channels = input_tensor.shape[1]
    out_channels = weight_tensor.shape[0]
    
    # Create output tensor
    output_shape = (batch_size, out_channels, height, width)
    output_tensor = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Determine grid
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (height + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )
    
    # Launch kernel
    if kernel_size == 1:
        conv2d_kernel[grid](
            input_tensor.data_ptr(),
            weight_tensor.data_ptr(),
            bias_tensor.data_ptr() if bias_tensor is not None else None,
            output_tensor.data_ptr(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            pad,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS=1,
        )
    else:
        # 3x3 kernel
        conv2d_kernel[grid](
            input_tensor.data_ptr(),
            weight_tensor.data_ptr(),
            bias_tensor.data_ptr() if bias_tensor is not None else None,
            output_tensor.data_ptr(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            pad,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS=1,
        )
    
    return output_tensor


def triton_relu(x):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x.data_ptr(), x.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        
        # Replace Conv2d with Triton kernels
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = triton_relu
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = triton_relu
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = triton_relu
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)