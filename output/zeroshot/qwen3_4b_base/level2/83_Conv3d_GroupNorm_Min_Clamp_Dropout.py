import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, d, h, w)
    output_ptr,  # pointer to output tensor (batch, out_channels, d, h, w)
    input_shape,  # (batch, in_channels, d, h, w)
    output_shape,  # (batch, out_channels, d, h, w)
    kernel,  # (out_channels, in_channels, d_k, h_k, w_k)
    stride,  # (stride_d, stride_h, stride_w)
    padding,  # (pad_d, pad_h, pad_w)
    dilation,  # (dil_d, dil_h, dil_w)
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Define block indices
    batch = tl.program_id(0)
    out_channel = tl.program_id(1)
    
    # Get input and output dimensions
    batch_size, in_channels, d_in, h_in, w_in = input_shape
    out_channels, _, d_k, h_k, w_k = kernel.shape
    d_out = (d_in + 2 * padding[0] - dilation[0] * (d_k - 1) - 1) // stride[0] + 1
    h_out = (h_in + 2 * padding[1] - dilation[1] * (h_k - 1) - 1) // stride[1] + 1
    w_out = (w_in + 2 * padding[2] - dilation[2] * (w_k - 1) - 1) // stride[2] + 1

    # Compute output indices
    d_out_idx = tl.arange(0, d_out)
    h_out_idx = tl.arange(0, h_out)
    w_out_idx = tl.arange(0, w_out)

    # Each thread computes one output element
    d_out_idx = tl.program_id(2)
    h_out_idx = tl.program_id(3)
    w_out_idx = tl.program_id(4)

    # Load output offset
    out_idx = batch * out_channels + out_channel
    out_offset = out_idx * d_out * h_out * w_out + d_out_idx * h_out * w_out + h_out_idx * w_out + w_out_idx

    # Compute input indices
    d_in_idx = d_out_idx * stride[0] - padding[0]
    h_in_idx = h_out_idx * stride[1] - padding[1]
    w_in_idx = w_out_idx * stride[2] - padding[2]

    # Compute valid input indices with dilation
    d_in_offset = d_in_idx + tl.arange(0, d_k) * dilation[0]
    h_in_offset = h_in_idx + tl.arange(0, h_k) * dilation[1]
    w_in_offset = w_in_idx + tl.arange(0, w_k) * dilation[2]

    # Define input and kernel indices
    input_offset = batch * in_channels * d_in * h_in * w_in
    kernel_offset = out_channel * in_channels * d_k * h_k * w_k

    # Load input and kernel values
    input_vals = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    kernel_vals = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Use shared memory for kernel (reduced kernel size)
    shared_kernel = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Load kernel values
    kernel_idx = tl.arange(0, BLOCK_SIZE)
    kernel_idx = kernel_idx % (d_k * h_k * w_k)
    d_k_idx = kernel_idx // (h_k * w_k)
    h_k_idx = (kernel_idx % (h_k * w_k)) // w_k
    w_k_idx = kernel_idx % w_k

    # Load kernel values into shared memory
    kernel_offset = out_channel * in_channels * d_k * h_k * w_k + d_k_idx * h_k * w_k + h_k_idx * w_k + w_k_idx
    kernel_vals = tl.load(kernel + kernel_offset, mask=kernel_idx < d_k * h_k * w_k, other=0.0)

    # Compute input indices
    input_idx = tl.arange(0, BLOCK_SIZE)
    input_idx = input_idx % (in_channels * d_in * h_in * w_in)
    in_channel_idx = input_idx // (d_in * h_in * w_in)
    d_in_idx = (input_idx % (d_in * h_in * w_in)) // (h_in * w_in)
    h_in_idx = (input_idx % (h_in * w_in)) // w_in
    w_in_idx = input_idx % w_in

    # Load input values
    input_vals = tl.load(input_ptr + input_offset + in_channel_idx * d_in * h_in * w_in + d_in_idx * h_in * w_in + h_in_idx * w_in + w_in_idx, mask=input_idx < in_channels * d_in * h_in * w_in, other=0.0)

    # Compute output
    output_val = tl.zeros(1, dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            for k in range(BLOCK_SIZE):
                d_in_idx = d_in_idx + i
                h_in_idx = h_in_idx + j
                w_in_idx = w_in_idx + k
                if (d_in_idx >= 0 and d_in_idx < d_in and
                    h_in_idx >= 0 and h_in_idx < h_in and
                    w_in_idx >= 0 and w_in_idx < w_in):
                    input_val = input_vals[i, j, k]
                    kernel_val = kernel_vals[i, j, k]
                    output_val += input_val * kernel_val

    # Store output
    tl.store(output_ptr + out_offset, output_val, mask=(d_out_idx < d_out) & (h_out_idx < h_out) & (w_out_idx < w_out))


@triton.jit
def group_norm_kernel(
    x_ptr,  # input tensor (batch, channels, d, h, w)
    y_ptr,  # output tensor (batch, channels, d, h, w)
    channels,  # number of channels
    group_size,  # group size
    eps,  # epsilon for stability
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    channel = tl.program_id(1)
    
    # Compute output dimensions
    d, h, w = 16, 64, 64
    total_channels = channels
    group_count = total_channels // group_size
    
    # Compute which group this thread belongs to
    group_idx = channel // group_size
    channel_in_group = channel % group_size
    
    # Compute input indices
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d * h * w
    input_idx = offsets + batch * d * h * w
    input_vals = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
    
    # Compute group mean and variance
    group_mean = tl.zeros(1, dtype=tl.float32)
    group_var = tl.zeros(1, dtype=tl.float32)
    
    # Reduce over spatial dimensions
    for i in range(BLOCK_SIZE):
        if mask[i]:
            group_mean += input_vals[i]
    group_mean = group_mean / tl.sum(mask)
    
    for i in range(BLOCK_SIZE):
        if mask[i]:
            diff = input_vals[i] - group_mean
            group_var += diff * diff
    group_var = group_var / tl.sum(mask)
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(group_var + eps)
    norm_val = (input_vals - group_mean) * inv_std
    
    # Store output
    tl.store(y_ptr + input_idx, norm_val, mask=mask)


@triton.jit
def min_clamp_kernel(
    x_ptr,  # input tensor (batch, out_channels, d, h, w)
    y_ptr,  # output tensor (batch, out_channels, d, h, w)
    min_val,  # minimum value
    max_val,  # maximum value
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    out_channel = tl.program_id(1)
    d_out = 16
    h_out = 64
    w_out = 64
    
    # Compute output indices
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_out * h_out * w_out
    out_idx = batch * out_channel * d_out * h_out * w_out + out_channel * d_out * h_out * w_out + offsets
    
    # Load input
    x_val = tl.load(x_ptr + out_idx, mask=mask, other=0.0)
    
    # Apply min and clamp
    min_val = tl.load(min_val)
    max_val = tl.load(max_val)
    x_clamped = tl.where(x_val < min_val, min_val, tl.where(x_val > max_val, max_val, x_val))
    
    # Store output
    tl.store(y_ptr + out_idx, x_clamped, mask=mask)


@triton.jit
def dropout_kernel(
    x_ptr,  # input tensor (batch, out_channels, d, h, w)
    y_ptr,  # output tensor (batch, out_channels, d, h, w)
    dropout_p,  # dropout probability
    mask_ptr,  # pointer to dropout mask
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    out_channel = tl.program_id(1)
    d_out = 16
    h_out = 64
    w_out = 64
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_out * h_out * w_out
    out_idx = batch * out_channel * d_out * h_out * w_out + out_channel * d_out * h_out * w_out + offsets
    
    # Load input
    x_val = tl.load(x_ptr + out_idx, mask=mask, other=0.0)
    
    # Generate dropout mask (random)
    mask_val = tl.load(mask_ptr + out_idx, mask=mask, other=1.0)
    dropout_mask = tl.where(tl.rand() < dropout_p, 0.0, 1.0)
    
    # Apply dropout
    y_val = x_val * dropout_mask
    
    # Store output
    tl.store(y_ptr + out_idx, y_val, mask=mask)


def triton_conv3d(
    input_tensor,
    kernel,
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    dilation=(1, 1, 1),
    groups=8,
):
    batch_size, in_channels, d_in, h_in, w_in = input_tensor.shape
    out_channels, _, d_k, h_k, w_k = kernel.shape
    d_out = (d_in + 2 * padding[0] - dilation[0] * (d_k - 1) - 1) // stride[0] + 1
    h_out = (h_in + 2 * padding[1] - dilation[1] * (h_k - 1) - 1) // stride[1] + 1
    w_out = (w_in + 2 * padding[2] - dilation[2] * (w_k - 1) - 1) // stride[2] + 1
    
    output_shape = (batch_size, out_channels, d_out, h_out, w_out)
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((batch_size, out_channels, d_out, h_out, w_out),)
    
    conv3d_kernel[grid](
        input_tensor.data_ptr(),
        output.data_ptr(),
        input_tensor.shape,
        output_shape,
        kernel.data_ptr(),
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=groups
    )
    return output


def triton_group_norm(
    x,
    num_groups,
    eps=1e-5,
):
    batch_size, channels, d, h, w = x.shape
    y = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((batch_size, channels),)
    
    group_norm_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        channels,
        num_groups,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def triton_min_clamp(
    x,
    min_val,
    max_val,
):
    y = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((x.shape[0], x.shape[1]),)
    
    min_clamp_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        min_val,
        max_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def triton_dropout(
    x,
    p,
    training=True,
):
    if not training:
        return x
    y = torch.empty_like(x)
    mask = torch.rand_like(x, device=x.device)
    mask = (mask < (1 - p)).float()
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((x.shape[0], x.shape[1]),)
    
    dropout_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        p,
        mask.data_ptr(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        # Initialize kernel
        self.kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
        
        # Define padding and stride
        self.padding = (1, 1, 1)
        self.stride = (1, 1, 1)
        self.dilation = (1, 1, 1)
        
    def forward(self, x):
        # 3D Convolution
        x = triton_conv3d(x, self.kernel, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        
        # Group Normalization
        x = triton_group_norm(x, self.groups, eps=1e-5)
        
        # Min and Clamp
        x = triton_min_clamp(x, self.min_value, self.max_value)
        
        # Dropout
        x = triton_dropout(x, self.dropout_p, training=True)
        
        return x