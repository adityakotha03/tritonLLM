import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, length)
    weight_ptr,  # Pointer to weight tensor (out_channels, in_channels // groups, kernel_size)
    bias_ptr,  # Pointer to bias tensor (out_channels)
    output_ptr,  # Pointer to output tensor (batch, out_channels, length_out)
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the output length
    out_length = (input_ptr.shape[2] + 2 * padding - kernel_size) // stride + 1

    # Compute the thread ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Compute the block start and end indices
    block_start = block_idx
    block_end = block_start + BLOCK_SIZE
    # Compute the range of indices for the current block
    offsets = tl.arange(0, BLOCK_SIZE)

    # Compute the output index for this block
    out_idx = block_start // out_channels
    out_channel = block_start % out_channels

    # Compute the input indices for this block
    in_channel = out_channel * groups
    in_channel_group = out_channel % groups

    # Compute the input offset
    input_offset = in_channel_group * in_channels // groups
    input_offset += (block_start // out_channels) * in_channels
    input_offset += block_start % out_channels * in_channels // groups

    # Load weights
    weight_offset = out_channel * in_channels // groups * kernel_size
    weight_offset += in_channel_group * kernel_size
    weight = tl.load(weight_ptr + weight_offset + tl.arange(0, kernel_size), other=0.0)

    # Compute the input start and end indices
    input_start = (block_start // out_channels) * stride - padding
    input_end = input_start + kernel_size * stride

    # Compute the output start and end indices
    output_start = (block_start // out_channels) // out_channels
    output_end = output_start + out_length

    # Compute the input and output indices for this block
    input_indices = input_start + tl.arange(0, kernel_size) * stride
    output_indices = output_start + tl.arange(0, BLOCK_SIZE)

    # Compute the input and output offsets
    input_offsets = input_indices + input_offset
    output_offsets = output_indices + out_channel * out_length

    # Load input data
    input_data = tl.load(input_ptr + input_offsets, mask=input_indices < input_start + kernel_size * stride, other=0.0)

    # Perform the convolution
    output = tl.dot(input_data, weight)

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_channel, other=0.0)
        output += bias

    # Store the result
    tl.store(output_ptr + output_offsets, output, mask=output_indices < out_length)


def triton_conv1d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and (bias.is_cuda if bias is not None else True), "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Compute the output length
    out_length = (input.size(2) + 2 * padding - weight.size(2)) // stride + 1
    output = torch.empty((input.size(0), weight.size(0), out_length), device=input.device, dtype=input.dtype)

    # Determine the number of blocks needed
    n_elements = input.size(0) * weight.size(0) * out_length
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1d_kernel[grid](input, weight, bias, output, input.size(0), input.size(1), weight.size(0), weight.size(2), stride, padding, groups, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Create weight and bias tensors
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            bias = None

        # Register the parameters
        self.register_parameter("weight", weight)
        if self.bias:
            self.register_parameter("bias", bias)

        # Perform the convolution
        return triton_conv1d(x, weight, bias, self.stride, self.padding, self.groups)