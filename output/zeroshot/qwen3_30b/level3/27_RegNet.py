import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triton_max_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    out_height,
    out_width,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Calculate the output spatial coordinates
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Calculate the starting position in the output
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    c = pid_c

    # Calculate the input region for this output element (2x2 kernel)
    in_h_start = out_h * 2
    in_w_start = out_w * 2

    # Load the 2x2 patch and compute the maximum
    max_val = -float('inf')
    for kh in range(2):
        for kw in range(2):
            h_idx = in_h_start + kh
            w_idx = in_w_start + kw
            # Check bounds
            valid = (h_idx < height) & (w_idx < width)
            if valid:
                val = tl.load(x_ptr + c * height * width + h_idx * width + w_idx)
                max_val = tl.maximum(max_val, val)

    # Store the result
    out_idx = c * out_height * out_width + out_h * out_width + out_w
    tl.store(out_ptr + out_idx, max_val)


def triton_max_pool(x: torch.Tensor, kernel_size=2, stride=2):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    out_height = (height + stride - 1) // stride
    out_width = (width + stride - 1) // stride

    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Determine block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    # We need to tile the output channels
    BLOCK_C = 16

    # Calculate grid dimensions
    grid_h = (out_height + BLOCK_H - 1) // BLOCK_H
    grid_w = (out_width + BLOCK_W - 1) // BLOCK_W
    grid_c = (channels + BLOCK_C - 1) // BLOCK_C

    # Launch the kernel
    triton_max_pool_kernel[grid_h, grid_w, grid_c](
        x,
        out,
        batch_size,
        channels,
        height,
        width,
        out_height,
        out_width,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    return out


@triton.jit
def triton_global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of output channels
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < channels

    # Calculate total elements for averaging
    total_elements = height * width

    # Accumulate the sum for each channel
    sum_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for h in range(height):
        for w in range(width):
            # Load the current spatial position for all channels in the block
            offsets = h * width + w
            channel_offsets = offsets + tl.arange(0, channels) * height * width
            vals = tl.load(x_ptr + channel_offsets, mask=mask, other=0.0)
            sum_vals += vals

    # Average
    avg_vals = sum_vals / total_elements

    # Store the result
    out_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_offsets, avg_vals, mask=mask)


def triton_global_avg_pool(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    batch_size, channels, height, width = x.shape

    # Prepare output tensor
    out = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)

    # Determine block size
    BLOCK_SIZE = 128

    # Calculate number of blocks
    num_blocks = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    triton_global_avg_pool_kernel[num_blocks](
        x,
        out,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def triton_linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of output features
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_features

    # Load weights for the current block of output features
    w_offsets = offsets * in_features + tl.arange(0, in_features)
    w_vals = tl.load(w_ptr + w_offsets, mask=mask[:, None], other=0.0)

    # Accumulate the dot product for each batch
    sum_vals = tl.zeros((batch_size, BLOCK_SIZE), dtype=tl.float32)
    for i in range(in_features):
        x_vals = tl.load(x_ptr + i * batch_size + tl.arange(0, batch_size), mask=None)
        sum_vals += x_vals[:, None] * w_vals[None, :]

    # Store the result
    out_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_offsets, sum_vals, mask=tl.arange(0, batch_size)[:, None] * mask[None, :])


def triton_linear(x: torch.Tensor, w: torch.Tensor):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Prepare output tensor
    out = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)

    # Determine block size
    BLOCK_SIZE = 128

    # Calculate number of blocks
    num_blocks = (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    triton_linear_kernel[num_blocks](
        x,
        w,
        out,
        batch_size,
        in_features,
        out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super().__init__()
        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)
    
    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Apply feature extractor with PyTorch ops for convolutions and ReLU
        x = self.feature_extractor(x)
        
        # Replace global average pooling with Triton kernel
        x = triton_global_avg_pool(x)
        
        # Replace linear layer with Triton kernel
        x = triton_linear(x, self.fc.weight)
        
        # Add bias if present
        if self.fc.bias is not None:
            x = x + self.fc.bias

        return x