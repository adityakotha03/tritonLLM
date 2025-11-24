import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    output_channels,  # Number of output channels
    kernel_size,  # Kernel size (assumed square)
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_id = pid
    # Compute the block position in the output
    block_row = block_id // (output_channels // BLOCK_SIZE)
    block_col = block_id % (output_channels // BLOCK_SIZE)
    # Compute the output position
    output_row = block_row * BLOCK_SIZE
    output_col = block_col * BLOCK_SIZE
    # Compute the input position
    input_row = output_row * stride - padding
    input_col = output_col * stride - padding
    # Compute the range of threads in the block
    thread_id = tl.program_id(1)
    # Compute the offset in the block
    offset = thread_id
    # Compute the input and output indices
    input_idx = (input_row * input_channels + input_col) * input_channels + offset
    output_idx = (output_row * output_channels + output_col) * output_channels + offset
    # Load weights
    weights = tl.load(weight_ptr + offset, mask=offset < output_channels, other=0.0)
    # Load input
    input_val = tl.load(input_ptr + input_idx, mask=input_idx < input_channels, other=0.0)
    # Compute the convolution
    output_val = tl.dot(input_val, weights)
    # Store output
    tl.store(output_ptr + output_idx, output_val)


@triton.jit
def batchnorm_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_id = pid
    # Compute the block position in the input
    block_row = block_id // (input_channels // BLOCK_SIZE)
    block_col = block_id % (input_channels // BLOCK_SIZE)
    # Compute the input and output indices
    input_idx = block_row * BLOCK_SIZE + block_col
    output_idx = block_row * BLOCK_SIZE + block_col
    # Load input
    input_val = tl.load(input_ptr + input_idx, mask=input_idx < input_channels, other=0.0)
    # Load weight and bias
    weight = tl.load(weight_ptr + input_idx, mask=input_idx < input_channels, other=0.0)
    bias = tl.load(bias_ptr + input_idx, mask=input_idx < input_channels, other=0.0)
    # Compute batchnorm
    output_val = (input_val - 0.0) * weight + bias
    # Store output
    tl.store(output_ptr + output_idx, output_val)


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_id = pid
    # Compute the block position in the input
    block_row = block_id // (input_channels // BLOCK_SIZE)
    block_col = block_id % (input_channels // BLOCK_SIZE)
    # Compute the input and output indices
    input_idx = block_row * BLOCK_SIZE + block_col
    output_idx = block_row * BLOCK_SIZE + block_col
    # Load input
    input_val = tl.load(input_ptr + input_idx, mask=input_idx < input_channels, other=0.0)
    # Compute ReLU
    output_val = tl.maximum(input_val, 0.0)
    # Store output
    tl.store(output_ptr + output_idx, output_val)


@triton.jit
def maxpool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    kernel_size,  # Kernel size (assumed square)
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_id = pid
    # Compute the block position in the output
    block_row = block_id // (input_channels // BLOCK_SIZE)
    block_col = block_id % (input_channels // BLOCK_SIZE)
    # Compute the input and output indices
    input_row = block_row * stride - padding
    input_col = block_col * stride - padding
    output_row = block_row
    output_col = block_col
    # Compute the range of threads in the block
    thread_id = tl.program_id(1)
    # Compute the offset in the block
    offset = thread_id
    # Compute the input indices
    input_idx = (input_row * input_channels + input_col) * input_channels + offset
    # Load input
    input_val = tl.load(input_ptr + input_idx, mask=input_idx < input_channels, other=0.0)
    # Compute max
    output_val = tl.max(input_val)
    # Store output
    tl.store(output_ptr + (output_row * input_channels + output_col), output_val)


def triton_conv2d(input, weight, input_channels, output_channels, kernel_size, stride, padding):
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    # Prepare output tensor
    output = torch.empty((input.size(0), output_channels, input.size(2) // stride, input.size(3) // stride), device=input.device)
    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_channels, output_channels, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_batchnorm(input, weight, bias, input_channels):
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    # Prepare output tensor
    output = torch.empty_like(input)
    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batchnorm_kernel[grid](input, weight, bias, output, input_channels, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_relu(input, input_channels):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    # Prepare output tensor
    output = torch.empty_like(input)
    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    relu_kernel[grid](input, output, input_channels, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_maxpool2d(input, input_channels, kernel_size, stride, padding):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    # Prepare output tensor
    output = torch.empty((input.size(0), input_channels, input.size(2) // stride, input.size(3) // stride), device=input.device)
    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    maxpool2d_kernel[grid](input, output, input_channels, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()
        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels

        # Construct the stages with their respective blocks
        for i in range(stages):
            # Create a custom block with conv2d, batchnorm, and relu
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]

        self.feature_extractor = nn.Sequential(*layers)

        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def _make_stage(self, in_channels, out_channels):
        """
        Creates a custom block for each stage with optimized Triton kernels.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return: nn.Sequential block with optimized Triton operations
        """
        return nn.Sequential(
            self._conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            self._batchnorm(out_channels),
            self._relu(),
            self._conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            self._batchnorm(out_channels),
            self._relu(),
            self._maxpool2d(kernel_size=2, stride=2, padding=0)
        )

    def _conv2d(self, in_channels, out_channels, kernel_size, stride, padding):
        weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        bias = nn.Parameter(torch.randn(out_channels))
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode='zeros')

    def _batchnorm(self, channels):
        weight = nn.Parameter(torch.randn(channels))
        bias = nn.Parameter(torch.randn(channels))
        return nn.BatchNorm2d(channels)

    def _relu(self):
        return nn.ReLU()

    def _maxpool2d(self, kernel_size, stride, padding):
        return nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        """
        Forward pass through the RegNet model with optimized Triton kernels.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x