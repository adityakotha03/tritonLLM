import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,       # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,      # pointer to weight tensor (out_channels, in_channels, kh, kw)
    output_ptr,      # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Define grid dimensions
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Compute output dimensions
    H_out = (height - padding_h - padding_h + output_padding_h) // stride_h + 1
    W_out = (width - padding_w - padding_w + output_padding_w) // stride_w + 1
    
    # For each output channel, compute the output spatial coordinates
    # We process one output channel at a time, and one batch at a time
    # Each block processes a portion of the output spatial grid
    
    # Define the current output position
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)
    
    # Only process valid output positions
    h_out = h_out * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w_out = w_out * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create valid range for output indices
    h_out_valid = h_out < H_out
    w_out_valid = w_out < W_out
    mask = h_out_valid & w_out_valid
    
    # Compute the corresponding input spatial coordinates
    # For transposed convolution: output (h_out, w_out) maps to input (h_in, w_in)
    h_in = (h_out * stride_h) - padding_h
    w_in = (w_out * stride_w) - padding_w
    
    # Clamp to valid input range
    h_in = tl.maximum(h_in, 0)
    w_in = tl.maximum(w_in, 0)
    h_in = tl.minimum(h_in, height - 1)
    w_in = tl.minimum(w_in, width - 1)
    
    # Compute input and weight indices
    # For each group, process the corresponding in_channels and out_channels
    group_idx = tl.arange(0, groups)
    group_size = in_channels // groups
    
    # Accumulate output across input channels
    out = tl.zeros((1,), dtype=tl.float32)
    
    # Process each group
    for g in group_idx:
        # Current input channel
        in_channel = g * group_size + tl.arange(0, group_size)
        
        # Load input patch (small window around (h_in, w_in))
        # We assume input is (batch, in_channels, H, W)
        # So input_ptr + batch_idx * in_channels * H * W + in_channel * H * W + h_in * W + w_in
        input_offset = (batch_idx * in_channels * height * width +
                        in_channel * height * width +
                        h_in * width + w_in)
        
        # Load input value
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Load weight (out_channel_idx, in_channel, kh, kw)
        # weight_ptr + out_channel_idx * in_channels * kh * kw + in_channel * kh * kw + kh_offset * kw + kw_offset
        kh_offset = tl.arange(0, kh)
        kw_offset = tl.arange(0, kw)
        kh, kw = kh, kw
        
        # Compute the input spatial coordinates for each weight
        # For transposed convolution: weight at (kh, kw) maps to input at (h_in - kh, w_in - kw)
        # But we need to compute the valid input positions
        # We are doing a direct convolution with the transposed kernel
        # So we compute the input positions as (h_in - kh, w_in - kw)
        h_in_kh = h_in - kh
        w_in_kw = w_in - kw
        
        # Clamp to valid input range
        h_in_kh = tl.maximum(h_in_kh, 0)
        w_in_kw = tl.maximum(w_in_kw, 0)
        h_in_kh = tl.minimum(h_in_kh, height - 1)
        w_in_kw = tl.minimum(w_in_kw, width - 1)
        
        # Compute weight indices
        weight_offset = (out_channel_idx * in_channels * kh * kw +
                         in_channel * kh * kw +
                         kh_offset * kw + kw_offset)
        
        # Load weight values
        weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Accumulate output
        out += input_val * weight_val
        
    # Store output
    output_offset = (batch_idx * out_channels * H_out * W_out +
                     out_channel_idx * H_out * W_out +
                     h_out * W_out + w_out)
    tl.store(output_ptr + output_offset, out, mask=mask)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    bias: bool = False,
) -> torch.Tensor:
    """
    Custom Triton kernel for transposed 2D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kh, kw = weight.shape
    
    # Compute output dimensions
    H_out = (height - padding - padding + output_padding) // stride + 1
    W_out = (width - padding - padding + output_padding) // stride + 1
    
    # Output tensor
    output = torch.empty((batch_size, out_channels, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block size (power of 2)
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (H_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )
    
    # Launch kernel
    conv_transpose2d_kernel[
        grid,
        (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    ](
        x.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        kh,
        kw,
        stride,
        stride,
        padding,
        padding,
        output_padding,
        output_padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Pre-allocate weight tensor
        kh, kw = kernel_size
        self.weight = torch.randn(out_channels, in_channels, kh, kw, dtype=torch.float16, device="cuda")
        
        # Optionally add bias
        if bias:
            self.bias = torch.randn(out_channels, dtype=torch.float16, device="cuda")
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom Triton kernel.
        """
        # Ensure inputs are on GPU and contiguous
        x = x.contiguous()
        
        # Apply transposed convolution via Triton kernel
        output = triton_conv_transpose2d(
            x, self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            bias=self.bias is not None
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            
        return output