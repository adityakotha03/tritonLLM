import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transpose_conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one output element
    # Compute the thread's output index
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (batch_size * out_channels * (depth_out + 2 * output_padding) * (height_out + 2 * output_padding) * (width_out + 2 * output_padding))

    # Compute the corresponding input indices
    out_idx = offset
    out_b = out_idx // ((depth_out + 2 * output_padding) * (height_out + 2 * output_padding) * (width_out + 2 * output_padding))
    out_c = (out_idx % ((depth_out + 2 * output_padding) * (height_out + 2 * output_padding) * (width_out + 2 * output_padding))) // ((height_out + 2 * output_padding) * (width_out + 2 * output_padding))
    out_d = (out_idx % ((height_out + 2 * output_padding) * (width_out + 2 * output_padding))) // (width_out + 2 * output_padding)
    out_h = (out_idx % (width_out + 2 * output_padding)) // (width_out + 2 * output_padding)
    out_w = out_idx % (width_out + 2 * output_padding)

    # Compute input indices
    in_d = out_d - (output_padding - padding) + (kernel_size - 1) * stride
    in_h = out_h - (output_padding - padding) + (kernel_size - 1) * stride
    in_w = out_w - (output_padding - padding) + (kernel_size - 1) * stride

    # Check if input indices are valid
    valid = (in_d >= 0) & (in_d < depth_in) & (in_h >= 0) & (in_h < height_in) & (in_w >= 0) & (in_w < width_in)

    # Initialize output
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over input channels
    for in_c in range(in_channels):
        # Compute weight indices
        weight_idx = out_c * in_channels * kernel_size * kernel_size * kernel_size + in_c * kernel_size * kernel_size * kernel_size
        weight_d = tl.arange(0, kernel_size)
        weight_h = tl.arange(0, kernel_size)
        weight_w = tl.arange(0, kernel_size)

        # Compute input indices for each weight
        in_d = out_d - (output_padding - padding) + (kernel_size - 1) * stride - weight_d
        in_h = out_h - (output_padding - padding) + (kernel_size - 1) * stride - weight_h
        in_w = out_w - (output_padding - padding) + (kernel_size - 1) * stride - weight_w

        # Check if input indices are valid
        valid = (in_d >= 0) & (in_d < depth_in) & (in_h >= 0) & (in_h < height_in) & (in_w >= 0) & (in_w < width_in)

        # Load input and weight
        input_val = tl.load(input_ptr + (out_b * in_channels * depth_in * height_in * width_in) + (in_c * depth_in * height_in * width_in) + (in_d * height_in * width_in) + (in_h * width_in) + in_w, mask=valid, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx + (weight_d * height_in * width_in) + (weight_h * width_in) + weight_w, mask=valid, other=0.0)

        # Multiply and accumulate
        out += input_val * weight_val

    # Store the result
    tl.store(output_ptr + offset, out, mask=mask)


def triton_transpose_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
):
    # Calculate output dimensions
    depth_in = input.size(2)
    height_in = input.size(3)
    width_in = input.size(4)
    depth_out = (depth_in - 1) * stride + kernel_size - 2 * padding + output_padding
    height_out = (height_in - 1) * stride + kernel_size - 2 * padding + output_padding
    width_out = (width_in - 1) * stride + kernel_size - 2 * padding + output_padding

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, depth_out, height_out, width_out), device=input.device, dtype=input.dtype)

    # Determine block size
    BLOCK_SIZE = 128

    # Launch kernel
    grid = lambda meta: ((batch_size * out_channels * depth_out * height_out * width_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    transpose_conv3d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get input dimensions
        batch_size = x.size(0)
        depth_in = x.size(2)
        height_in = x.size(3)
        width_in = x.size(4)

        # Calculate output dimensions
        depth_out = (depth_in - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        height_out = (height_in - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        width_out = (width_in - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding

        # Initialize weight and bias
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size, self.kernel_size, device=x.device, dtype=x.dtype))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels, device=x.device, dtype=x.dtype))
        else:
            bias = None

        # Perform transpose convolution using Triton kernel
        output = triton_transpose_conv3d(x, weight, bias, batch_size, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding)

        # Apply bias if needed
        if self.bias:
            output += bias.view(1, -1, 1, 1, 1)

        return output