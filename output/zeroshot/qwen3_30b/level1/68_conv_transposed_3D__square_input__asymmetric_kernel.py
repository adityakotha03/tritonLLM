import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for transposed 3D convolution with operator fusion (conv_transpose + bias + activation)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,       # Input tensor pointer
    w_ptr,       # Weight tensor pointer
    b_ptr,       # Bias pointer (if bias is True)
    out_ptr,     # Output tensor pointer
    batch_size,  # Batch size
    in_channels, # Number of input channels
    out_channels,# Number of output channels
    depth,       # Input depth
    width,       # Input width
    height,      # Input height
    out_depth,   # Output depth
    out_width,   # Output width
    out_height,  # Output height
    kernel_depth,# Kernel depth
    kernel_width,# Kernel width
    kernel_height,# Kernel height
    stride_depth,# Stride in depth
    stride_width,# Stride in width
    stride_height,# Stride in height
    pad_depth,   # Padding in depth
    pad_width,   # Padding in width
    pad_height,  # Padding in height
    output_padding_depth,  # Output padding in depth
    output_padding_width,  # Output padding in width
    output_padding_height, # Output padding in height
    groups,      # Number of groups
    has_bias,    # Boolean flag for bias
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Shared memory for weight tiles and accumulator
    pid = tl.program_id(0)  # Block ID for output feature maps
    pid_batch = pid // (out_channels // groups)
    pid_outch = pid % (out_channels // groups)
    pid_group = pid // (out_channels // groups)
    pid_outch_in_group = pid % (out_channels // groups)
    
    # Get batch and output channel indices
    batch_idx = pid_batch
    outch_idx = pid_outch

    # Determine output tensor coordinate
    out_d = tl.load(tl.arange(0, TILE_SIZE) + out_depth * pid_outch_in_group)
    out_w = tl.load(tl.arange(0, TILE_SIZE) + out_width * pid_outch_in_group)
    out_h = tl.load(tl.arange(0, TILE_SIZE) + out_height * pid_outch_in_group)

    # Bounds check
    mask_d = out_d < out_depth
    mask_w = out_w < out_width
    mask_h = out_h < out_height
    mask = mask_d[:, None, None] & mask_w[None, :, None] & mask_h[None, None, :]

    # Compute input coordinates
    in_d = out_d * stride_depth - pad_depth + kernel_depth - 1
    in_w = out_w * stride_width - pad_width + kernel_width - 1
    in_h = out_h * stride_height - pad_height + kernel_height - 1

    # Offset input indices
    in_d_offset = in_d
    in_w_offset = in_w
    in_h_offset = in_h

    # Handle output padding
    in_d_offset += output_padding_depth
    in_w_offset += output_padding_width
    in_h_offset += output_padding_height

    # Compute input index: [batch, in_channels, in_d, in_w, in_h]
    in_offset = (batch_idx * in_channels + tl.arange(0, in_channels)[:, None, None, None]) * (depth * width * height) + \
                in_d_offset * (width * height) + \
                in_w_offset * (height) + \
                in_h_offset

    # Compute weight index: [out_channels, in_channels, kernel_depth, kernel_width, kernel_height]
    w_offset = (outch_idx * in_channels + tl.arange(0, in_channels)[:, None, None, None]) * (kernel_depth * kernel_width * kernel_height) + \
               tl.arange(0, kernel_depth)[:, None, None] * (kernel_width * kernel_height) + \
               tl.arange(0, kernel_width)[:, None] * (kernel_height) + \
               tl.arange(0, kernel_height)

    # Load input and weight
    x = tl.load(x_ptr + in_offset, mask=mask[None, :, :, :], other=0.0)
    w = tl.load(w_ptr + w_offset, mask=mask[None, :, :, :], other=0.0)

    # Compute output: convolution sum over input channels and kernel
    # Accumulate across in_channels, kernel_depth, kernel_width, kernel_height
    out = tl.sum(x * w, axis=(1, 2, 3, 4))

    # Add bias
    if has_bias:
        bias = tl.load(b_ptr + outch_idx, mask=mask[None, :, :], other=0.0)
        out = out + bias

    # Apply activation (e.g., ReLU)
    if ACTIVATION == 1:  # ReLU
        out = tl.max(out, 0)
    elif ACTIVATION == 2:  # Sigmoid
        out = 1 / (1 + tl.exp(-out))

    # Store output
    out_offset = (batch_idx * out_channels + outch_idx) * (out_depth * out_width * out_height) + \
                 out_d * (out_width * out_height) + \
                 out_w * (out_height) + \
                 out_h

    tl.store(out_ptr + out_offset, out, mask=mask)

# Triton wrapper for the transposed 3D convolution
def triton_conv_transpose3d(x, w, b=None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=True, activation='none'):
    """
    Custom Triton-based transposed 3D convolution.
    Supports bias and activation (ReLU, Sigmoid).
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    assert x.dim() == 5, "Input must be 5D: (batch, in_channels, depth, width, height)"
    assert w.dim() == 5, "Weights must be 5D: (out_channels, in_channels, kernel_depth, kernel_width, kernel_height)"
    
    batch_size, in_channels, depth, width, height = x.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = w.shape
    stride_depth, stride_width, stride_height = stride
    pad_depth, pad_width, pad_height = padding
    out_pad_depth, out_pad_width, out_pad_height = output_padding

    # Compute output spatial dimensions
    out_depth = (depth - 1) * stride_depth - 2 * pad_depth + kernel_depth + out_pad_depth
    out_width = (width - 1) * stride_width - 2 * pad_width + kernel_width + out_pad_width
    out_height = (height - 1) * stride_height - 2 * pad_height + kernel_height + out_pad_height

    # Ensure groups is valid
    assert out_channels % groups == 0, "out_channels must be divisible by groups"
    assert in_channels % groups == 0, "in_channels must be divisible by groups"

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_width, out_height, device=x.device, dtype=x.dtype)

    # Ensure contiguity
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous() if b is not None else None

    # Set up grid
    total_blocks = (batch_size * out_channels) // groups
    grid = lambda meta: (total_blocks,)

    # Determine activation mode
    activation_map = {'none': 0, 'relu': 1, 'sigmoid': 2}
    act_type = activation_map.get(activation, 0)

    # Launch kernel
    conv_transpose3d_kernel[grid](
        x_ptr=x, w_ptr=w, b_ptr=b, out_ptr=out,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        depth=depth, width=width, height=height,
        out_depth=out_depth, out_width=out_width, out_height=out_height,
        kernel_depth=kernel_depth, kernel_width=kernel_width, kernel_height=kernel_height,
        stride_depth=stride_depth, stride_width=stride_width, stride_height=stride_height,
        pad_depth=pad_depth, pad_width=pad_width, pad_height=pad_height,
        output_padding_depth=out_pad_depth, output_padding_width=out_pad_width, output_padding_height=out_pad_height,
        groups=groups, has_bias=int(bias), BLOCK_SIZE=128, TILE_SIZE=64, ACTIVATION=act_type
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False, activation: str = 'none'):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.activation = activation

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups, 
            bias=self.bias is not None, 
            activation=self.activation
        )