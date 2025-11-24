import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1x1_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height of input
    width,  # Width of input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Compute the position in the output
    out_h = (pid // width) % height
    out_w = (pid % width) % width
    out_c = pid % out_channels
    # Compute the input position
    in_c = tl.arange(0, in_channels)
    # Compute the offset for the input
    in_offset = tl.arange(0, BLOCK_SIZE)
    in_offset = in_offset + tl.arange(0, BLOCK_SIZE) * in_channels
    in_offset = in_offset + out_c * in_channels * height * width
    in_offset = in_offset + out_h * width * in_channels + out_w
    # Load input values
    input_vals = tl.load(input_ptr + in_offset, mask=in_offset < batch_size * in_channels * height * width, other=0.0)
    # Compute the output
    output = tl.dot(input_vals, weight_ptr + out_c * in_channels)
    # Store the output
    output_ptr = output_ptr + out_c * height * width + out_h * width + out_w
    tl.store(output_ptr, output)


@triton.jit
def conv3x3_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height of input
    width,  # Width of input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Compute the position in the output
    out_h = (pid // width) % height
    out_w = (pid % width) % width
    out_c = pid % out_channels
    # Compute the input positions (3x3 window)
    in_h = tl.arange(0, 3)
    in_w = tl.arange(0, 3)
    # Compute the input offset
    in_offset = (in_h * width + in_w) * in_channels + tl.arange(0, in_channels)
    in_offset = in_offset + out_c * in_channels * height * width
    in_offset = in_offset + out_h * width * in_channels + out_w
    # Load input values
    input_vals = tl.load(input_ptr + in_offset, mask=in_offset < batch_size * in_channels * height * width, other=0.0)
    # Compute the output
    output = tl.dot(input_vals, weight_ptr + out_c * in_channels * 9)
    # Store the output
    output_ptr = output_ptr + out_c * height * width + out_h * width + out_w
    tl.store(output_ptr, output)


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    height,  # Height of input
    width,  # Width of input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Compute the position in the output
    out_h = (pid // width) % height
    out_w = (pid % width) % width
    out_c = pid % in_channels
    # Compute the input offset
    in_offset = out_c * height * width + out_h * width + out_w
    # Load input values
    input_vals = tl.load(input_ptr + in_offset, mask=in_offset < batch_size * in_channels * height * width, other=0.0)
    # Compute the output
    output = tl.maximum(input_vals, 0.0)
    # Store the output
    output_ptr = output_ptr + out_c * height * width + out_h * width + out_w
    tl.store(output_ptr, output)


def triton_conv1x1(x: torch.Tensor, weight: torch.Tensor, out_channels: int, height: int, width: int):
    """
    Custom Triton kernel for 1x1 convolution with ReLU.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    # Prepare output tensor
    out = torch.empty((x.shape[0], out_channels, height, width), dtype=x.dtype, device=x.device)
    # Number of elements in the tensor
    n_elements = x.shape[0] * out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1x1_kernel[grid](x, weight, out, x.shape[0], x.shape[1], out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_conv3x3(x: torch.Tensor, weight: torch.Tensor, out_channels: int, height: int, width: int):
    """
    Custom Triton kernel for 3x3 convolution with ReLU.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    # Prepare output tensor
    out = torch.empty((x.shape[0], out_channels, height, width), dtype=x.dtype, device=x.device)
    # Number of elements in the tensor
    n_elements = x.shape[0] * out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3x3_kernel[grid](x, weight, out, x.shape[0], x.shape[1], out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_relu(x: torch.Tensor):
    """
    Custom Triton kernel for ReLU.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    relu_kernel[grid](x, out, x.shape[0], x.shape[1], x.shape[2], x.shape[3], BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(ModelNew, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_weight = self.squeeze.weight
        self.squeeze_bias = self.squeeze.bias

        self.expand1x1_weight = nn.Parameter(torch.randn(expand1x1_channels, squeeze_channels, 1, 1))
        self.expand1x1_bias = nn.Parameter(torch.randn(expand1x1_channels))

        self.expand3x3_weight = nn.Parameter(torch.randn(expand3x3_channels, squeeze_channels, 3, 3))
        self.expand3x3_bias = nn.Parameter(torch.randn(expand3x3_channels))

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        # Squeeze layer
        x = self.squeeze(x)
        x = triton_relu(x)

        # Expand 1x1 layer
        expand1x1 = triton_conv1x1(x, self.expand1x1_weight, self.expand1x1_channels, x.shape[2], x.shape[3])
        expand1x1 = triton_relu(expand1x1 + self.expand1x1_bias.view(1, -1, 1, 1))

        # Expand 3x3 layer
        expand3x3 = triton_conv3x3(x, self.expand3x3_weight, self.expand3x3_channels, x.shape[2], x.shape[3])
        expand3x3 = triton_relu(expand3x3 + self.expand3x3_bias.view(1, -1, 1, 1))

        return torch.cat([expand1x1, expand3x3], 1)