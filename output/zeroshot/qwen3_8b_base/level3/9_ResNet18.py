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
    # Get program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_start = pid * BLOCK_SIZE
    # Compute the block's starting position in the output
    block_output_start = block_start // (output_channels * output_channels)
    block_output_row = block_output_start % output_channels
    block_output_col = block_output_start // output_channels

    # Compute the corresponding input position
    input_row = block_output_row * stride - padding
    input_col = block_output_col * stride - padding

    # Compute the range of threads in the block
    offset = tl.arange(0, BLOCK_SIZE)
    # Compute the indices in the output
    output_idx = block_output_row * output_channels + block_output_col + offset
    # Compute the indices in the input
    input_idx = input_row * input_channels + input_col + offset

    # Load weights
    weight = tl.load(weight_ptr + offset, mask=offset < output_channels, other=0.0)

    # Accumulate the result
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            input_idx_i = input_row + i
            input_idx_j = input_col + j
            input_idx = input_idx_i * input_channels + input_idx_j + offset
            input_val = tl.load(input_ptr + input_idx, mask=input_idx < input_channels * input_channels, other=0.0)
            out += input_val * weight

    # Store the result
    tl.store(output_ptr + output_idx, out, mask=output_idx < output_channels * output_channels)

def triton_conv2d(input, weight, stride, padding):
    """
    This function wraps the Triton kernel call for convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Compute output dimensions
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    output = torch.zeros((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)

    # Number of elements per block
    num_elements_per_block = out_channels * out_channels * BLOCK_SIZE
    num_blocks = (out_channels * out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    conv2d_kernel[grid](input, weight, output, in_channels, out_channels, kernel_size, stride, padding, BLOCK_SIZE=128)
    return output

@triton.jit
def batchnorm2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    mean_ptr,  # Pointer to mean tensor
    rstd_ptr,  # Pointer to reciprocal standard deviation tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    height,  # Height of the input
    width,  # Width of the input
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_start = pid * BLOCK_SIZE
    # Compute the block's starting position in the output
    block_output_start = block_start // (input_channels * height * width)
    block_output_idx = block_start % (input_channels * height * width)

    # Compute the indices in the output
    output_idx = block_output_idx + offset
    # Compute the indices in the input
    input_idx = block_output_idx + offset

    # Load input value
    x = tl.load(input_ptr + input_idx, mask=input_idx < input_channels * height * width, other=0.0)

    # Load weight and bias
    w = tl.load(weight_ptr + offset, mask=offset < input_channels, other=0.0)
    b = tl.load(bias_ptr + offset, mask=offset < input_channels, other=0.0)

    # Compute normalization
    x = (x - mean) * rstd
    x = x * w + b

    # Store the result
    tl.store(output_ptr + output_idx, x, mask=output_idx < input_channels * height * width)

def triton_batchnorm2d(input, weight, bias, mean, rstd):
    """
    This function wraps the Triton kernel call for batch normalization.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and mean.is_cuda and rstd.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()

    output = torch.empty_like(input)

    # Number of elements per block
    num_elements_per_block = input_channels * height * width * BLOCK_SIZE
    num_blocks = (input_channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    batchnorm2d_kernel[grid](input, weight, bias, mean, rstd, output, input_channels, height, width, eps=1e-5, BLOCK_SIZE=128)
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x