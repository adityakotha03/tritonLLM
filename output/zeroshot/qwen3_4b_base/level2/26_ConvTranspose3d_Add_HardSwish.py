import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    input_ptr,           # Pointer to input tensor (batch, in_channels, D, H, W)
    output_ptr,          # Pointer to output tensor (batch, out_channels, D_out, H_out, W_out)
    weight_ptr,          # Pointer to weight tensor (out_channels, in_channels, d_k, h_k, w_k)
    bias_ptr,            # Pointer to bias tensor (out_channels)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    d_k: tl.constexpr,
    h_k: tl.constexpr,
    w_k: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    output_padding_d: tl.constexpr,
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block and thread indices
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    
    # Compute the output spatial indices
    d_out = tl.arange(0, D_out)
    h_out = tl.arange(0, H_out)
    w_out = tl.arange(0, W_out)
    
    # Compute the input spatial indices using stride and padding
    d_in = (d_out * stride_d) - padding_d
    h_in = (h_out * stride_h) - padding_h
    w_in = (w_out * stride_w) - padding_w
    
    # Create the full offset for the output tensor
    offset_d = d_out + output_padding_d
    offset_h = h_out + output_padding_h
    offset_w = w_out + output_padding_w
    
    # Compute the input spatial indices with padding
    d_in_start = d_in - padding_d
    h_in_start = h_in - padding_h
    w_in_start = w_in - padding_w
    
    # Compute the output offset for the current block
    d_out_idx = tl.program_id(0) * BLOCK_SIZE // D_out
    h_out_idx = (tl.program_id(0) * BLOCK_SIZE) % D_out
    w_out_idx = (tl.program_id(0) * BLOCK_SIZE) % H_out
    
    # Compute the block of input indices to process
    d_in_offset = tl.arange(0, BLOCK_SIZE)
    h_in_offset = tl.arange(0, BLOCK_SIZE)
    w_in_offset = tl.arange(0, BLOCK_SIZE)
    
    # Create the 3D index for input and output
    d_in_idx = d_in_offset + d_in_start
    h_in_idx = h_in_offset + h_in_start
    w_in_idx = w_in_offset + w_in_start
    
    # Create the mask to avoid out-of-bounds access
    d_in_mask = (d_in_idx >= 0) & (d_in_idx < D_in)
    h_in_mask = (h_in_idx >= 0) & (h_in_idx < H_in)
    w_in_mask = (w_in_idx >= 0) & (w_in_idx < W_in)
    
    # Combine masks
    mask = d_in_mask & h_in_mask & w_in_mask
    
    # Load input features
    input_batch = batch_idx
    input_channels = tl.arange(0, in_channels)
    input_d = d_in_idx
    input_h = h_in_idx
    input_w = w_in_idx
    
    # Load weights
    weight_d = tl.arange(0, d_k)
    weight_h = tl.arange(0, h_k)
    weight_w = tl.arange(0, w_k)
    
    # Compute the output value for each channel
    output_val = tl.zeros((out_channels,), dtype=tl.float32)
    
    # Iterate over spatial positions
    for i in range(BLOCK_SIZE):
        d_in_idx_i = d_in_idx[i]
        h_in_idx_i = h_in_idx[i]
        w_in_idx_i = w_in_idx[i]
        
        if not mask[i]:
            continue
            
        # Load input value
        input_val = tl.load(input_ptr + batch_idx * in_channels * D_in * H_in * W_in +
                           input_channels * D_in * H_in * W_in +
                           input_d[i] * H_in * W_in +
                           input_h[i] * W_in +
                           input_w[i],
                           mask=mask[i], other=0.0)
        
        # Load weights
        weight_val = tl.load(weight_ptr + out_channels * in_channels * d_k * h_k * w_k +
                             input_channels * d_k * h_k * w_k +
                             weight_d[i] * h_k * w_k +
                             weight_h[i] * w_k +
                             weight_w[i],
                             mask=tl.all(weight_d[i] < d_k) & tl.all(weight_h[i] < h_k) & tl.all(weight_w[i] < w_k),
                             other=0.0)
        
        # Accumulate output
        output_val += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + out_channels, mask=tl.arange(0, out_channels) < out_channels, other=0.0)
    output_val = output_val + bias_val
    
    # Store output
    output_offset = out_idx * D_out * H_out * W_out + d_out_idx * H_out * W_out + h_out_idx * W_out + w_out_idx
    tl.store(output_ptr + batch_idx * out_channels * D_out * H_out * W_out +
             output_offset,
             output_val, mask=mask)


@triton.jit
def add_and_hardswish_kernel(
    x_ptr,                # Pointer to transposed conv output
    add_input_ptr,        # Pointer to add_input tensor
    out_ptr,              # Pointer to output tensor
    batch_size: tl.constexpr,
    out_channels: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    
    # Define spatial indices
    d_out = tl.arange(0, D_out)
    h_out = tl.arange(0, H_out)
    w_out = tl.arange(0, W_out)
    
    # Compute output offset
    offset = out_idx * D_out * H_out * W_out + d_out * H_out * W_out + h_out * W_out + w_out
    
    # Load x and add_input
    x_val = tl.load(x_ptr + batch_idx * out_channels * D_out * H_out * W_out + offset, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=0.0)
    add_input_val = tl.load(add_input_ptr + batch_idx * out_channels * D_out * H_out * W_out + offset, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=0.0)
    
    # Add inputs
    sum_val = x_val + add_input_val
    
    # HardSwish activation: x * (x + 3) / 6 for x >= 0, and x * (x + 3) / 6 for x < 0
    # We can split into two parts
    pos_mask = sum_val >= 0.0
    neg_mask = sum_val < 0.0
    
    # Compute activation
    pos_val = sum_val * (sum_val + 3.0) / 6.0
    neg_val = sum_val * (sum_val + 3.0) / 6.0
    
    # Use conditional logic with masking
    out_val = tl.where(pos_mask, pos_val, neg_val)
    
    # Store result
    tl.store(out_ptr + batch_idx * out_channels * D_out * H_out * W_out + offset, out_val, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE)


def triton_conv_transpose_3d(
    input_tensor,
    weight_tensor,
    bias_tensor,
    batch_size,
    in_channels,
    out_channels,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    d_k,
    h_k,
    w_k,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    output_padding_d,
    output_padding_h,
    output_padding_w,
):
    """
    Custom Triton kernel for 3D transposed convolution.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    assert weight_tensor.is_cuda, "Weight tensor must be on CUDA."
    assert bias_tensor.is_cuda, "Bias tensor must be on CUDA."

    # Ensure contiguous
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    bias_tensor = bias_tensor.contiguous()

    # Output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Define grid
    grid = lambda meta: (
        (batch_size * out_channels * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        D_out * H_out * W_out // meta["BLOCK_SIZE"] + 1,
    )

    # Launch kernel
    conv_transpose_3d_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        d_k,
        h_k,
        w_k,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        BLOCK_SIZE=128,
    )
    return output_tensor


def triton_add_and_hardswish(
    x_tensor,
    add_input_tensor,
    out_tensor,
    batch_size,
    out_channels,
    D_out,
    H_out,
    W_out,
):
    """
    Custom Triton kernel for adding input and applying HardSwish activation.
    """
    assert x_tensor.is_cuda and add_input_tensor.is_cuda, "Tensors must be on CUDA."
    x_tensor = x_tensor.contiguous()
    add_input_tensor = add_input_tensor.contiguous()
    out_tensor = torch.empty_like(x_tensor)

    grid = lambda meta: (
        (batch_size * out_channels * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        D_out * H_out * W_out // meta["BLOCK_SIZE"] + 1,
    )

    add_and_hardswish_kernel[
        grid
    ](
        x_tensor.data_ptr(),
        add_input_tensor.data_ptr(),
        out_tensor.data_ptr(),
        batch_size,
        out_channels,
        D_out,
        H_out,
        W_out,
        BLOCK_SIZE=128,
    )
    return out_tensor


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # We will use custom kernels instead of nn.ConvTranspose3d and nn.functional.hardswish
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        
        # Weight and bias parameters are initialized in the forward pass
        # We assume the weights and bias are provided as inputs to forward
        # For now, we store them as parameters
        self.register_parameter("bias", nn.Parameter(torch.randn(bias_shape)))
        
        # We will compute the output dimensions based on the input
        self.d_k, self.h_k, self.w_k = kernel_size[0], kernel_size[1], kernel_size[2]
        self.stride_d, self.stride_h, self.stride_w = stride[0], stride[1], stride[2]
        self.padding_d, self.padding_h, self.padding_w = padding[0], padding[1], padding[2]
        self.output_padding_d, self.output_padding_h, self.output_padding_w = output_padding[0], output_padding[1], output_padding[2]
        
        # Precompute output dimensions
        self.D_out = (D_in := 16) * self.stride_d - self.padding_d + self.output_padding_d
        self.H_out = (H_in := 16) * self.stride_h - self.padding_h + self.output_padding_h
        self.W_out = (W_in := 16) * self.stride_w - self.padding_w + self.output_padding_w
        
    def forward(self, x, add_input):
        # x: (batch, in_channels, D, H, W)
        # add_input: (batch, out_channels, D_out, H_out, W_out)
        
        # Step 1: Apply 3D transposed convolution using custom kernel
        # We assume weight tensor is passed in or initialized
        # For now, we simulate it with a placeholder
        # In a real implementation, weight would be a parameter
        
        # Use custom kernel for transposed convolution
        weight = torch.randn(self.out_channels, self.in_channels, self.d_k, self.h_k, self.w_k).cuda()
        
        # Run transposed convolution
        conv_output = triton_conv_transpose_3d(
            x, weight, self.bias, self.in_channels, self.out_channels,
            self.d_k, self.h_k, self.w_k,
            self.stride_d, self.stride_h, self.stride_w,
            self.padding_d, self.padding_h, self.padding_w,
            self.output_padding_d, self.output_padding_h, self.output_padding_w
        )
        
        # Step 2: Add input tensor
        # Step 3: Apply HardSwish activation using custom kernel
        output = triton_add_and_hardswish(
            conv_output, add_input,
            torch.empty_like(conv_output),
            x.shape[0], self.out_channels, self.D_out, self.H_out, self.W_out
        )
        
        return output