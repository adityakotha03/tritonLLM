import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,      # Pointer to input tensor (batch, in_channels, depth, height, width)
    weight_ptr,     # Pointer to weight tensor (out_channels, in_channels, d_k, h_k, w_k)
    bias_ptr,       # Pointer to bias tensor (out_channels) - optional
    output_ptr,     # Pointer to output tensor (batch, out_channels, depth_out, height_out, width_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute the block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    depth_idx = tl.program_id(2)
    height_idx = tl.program_id(3)
    width_idx = tl.program_id(4)

    # Compute the actual block boundaries
    batch_start = batch_idx * batch_size
    out_channel_start = out_channel_idx * out_channels
    depth_start = depth_idx * (kernel_d + 2 * padding_d - 1) // stride_d
    height_start = height_idx * (kernel_h + 2 * padding_h - 1) // stride_h
    width_start = width_idx * (kernel_w + 2 * padding_w - 1) // stride_w

    # Define the range of indices for each dimension
    block_d = tl.arange(0, BLOCK_SIZE)
    block_h = tl.arange(0, BLOCK_SIZE_H)
    block_w = tl.arange(0, BLOCK_SIZE_W)

    # Compute the output dimensions
    depth_out = (depth_start + block_d) // stride_d
    height_out = (height_start + block_h) // stride_h
    width_out = (width_start + block_w) // stride_w

    # Compute the input and output indices
    input_depth = depth_start + block_d
    input_height = height_start + block_h
    input_width = width_start + block_w

    # Compute the output index for this block
    output_depth = (input_depth - padding_d) // stride_d
    output_height = (input_height - padding_h) // stride_h
    output_width = (input_width - padding_w) // stride_w

    # Create mask for valid indices
    valid_d = (input_depth >= 0) & (input_depth < input_depth.max())
    valid_h = (input_height >= 0) & (input_height < input_height.max())
    valid_w = (input_width >= 0) & (input_width < input_width.max())

    # Load input features
    input_offset = (batch_start * in_channels * depth * height * width +
                    (input_depth * height * width + input_height * width + input_width) * in_channels)
    input_values = tl.load(input_ptr + input_offset, mask=valid_d & valid_h & valid_w, other=0.0)

    # Load weights
    weight_offset = (out_channel_start * in_channels * kernel_d * kernel_h * kernel_w +
                     (block_d * kernel_h * kernel_w + block_h * kernel_w + block_w) * in_channels)
    weight_values = tl.load(weight_ptr + weight_offset, mask=valid_d & valid_h & valid_w, other=0.0)

    # Compute output for each channel
    output_values = tl.zeros((out_channels,), dtype=tl.float32)
    for i in range(out_channels):
        # Use fused computation with shared memory
        output_values = tl.sum(input_values * weight_values, axis=0)

    # Add bias if present
    if bias_ptr is not None:
        bias_offset = out_channel_start + i
        bias_val = tl.load(bias_ptr + bias_offset, mask=(i < out_channels), other=0.0)
        output_values += bias_val

    # Store output
    output_offset = (batch_start * out_channels * depth_out * height_out * width_out +
                     (output_depth * height_out * width_out + output_height * width_out + output_width) * out_channels)
    tl.store(output_ptr + output_offset, output_values, mask=valid_d & valid_h & valid_w)


@triton.jit
def conv3d_kernel_fused(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    depth_idx = tl.program_id(2)
    height_idx = tl.program_id(3)
    width_idx = tl.program_id(4)

    # Define local indices
    block_d = tl.arange(0, BLOCK_SIZE)
    block_h = tl.arange(0, BLOCK_SIZE_H)
    block_w = tl.arange(0, BLOCK_SIZE_W)

    # Compute input and output indices
    input_depth = depth_idx * (kernel_d + 2 * padding_d - 1) // stride_d + block_d
    input_height = height_idx * (kernel_h + 2 * padding_h - 1) // stride_h + block_h
    input_width = width_idx * (kernel_w + 2 * padding_w - 1) // stride_w + block_w

    # Compute output indices
    output_depth = (input_depth - padding_d) // stride_d
    output_height = (input_height - padding_h) // stride_h
    output_width = (input_width - padding_w) // stride_w

    # Compute valid mask
    valid_d = (input_depth >= 0) & (input_depth < input_depth.max())
    valid_h = (input_height >= 0) & (input_height < input_height.max())
    valid_w = (input_width >= 0) & (input_width < input_width.max())

    # Load input and weights
    input_offset = (batch_idx * in_channels * depth * height * width +
                    (input_depth * height * width + input_height * width + input_width) * in_channels)
    input_values = tl.load(input_ptr + input_offset, mask=valid_d & valid_h & valid_w, other=0.0)

    weight_offset = (out_channel_idx * in_channels * kernel_d * kernel_h * kernel_w +
                     (block_d * kernel_h * kernel_w + block_h * kernel_w + block_w) * in_channels)
    weight_values = tl.load(weight_ptr + weight_offset, mask=valid_d & valid_h & valid_w, other=0.0)

    # Perform convolution
    output_values = tl.zeros((out_channels,), dtype=tl.float32)
    for i in range(out_channels):
        output_values += tl.sum(input_values * weight_values, axis=0)

    # Add bias
    if bias_ptr is not None:
        bias_offset = out_channel_idx
        bias_val = tl.load(bias_ptr + bias_offset, mask=(i < out_channels), other=0.0)
        output_values += bias_val

    # Store output
    output_offset = (batch_idx * out_channels * depth_out * height_out * width_out +
                     (output_depth * height_out * width_out + output_height * width_out + output_width) * out_channels)
    tl.store(output_ptr + output_offset, output_values, mask=valid_d & valid_h & valid_w)


def triton_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    dilation: tuple = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """
    Custom 3D convolution using Triton kernels.
    """
    assert input.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, depth, height, width = input.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape

    # Define block sizes
    BLOCK_SIZE = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Grid dimensions
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (depth + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch kernel
    output = torch.empty_like(input)
    conv3d_kernel_fused[
        grid
    ](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_d=kernel_d,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_d=stride[0],
        stride_h=stride[1],
        stride_w=stride[2],
        padding_d=padding[0],
        padding_h=padding[1],
        padding_w=padding[2],
        dilation_d=dilation[0],
        dilation_h=dilation[1],
        dilation_w=dilation[2],
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)