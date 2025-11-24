import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

@triton.jit
def conv1x1_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    stride,  # Stride of the convolution
    padding,  # Padding of the convolution
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    height,  # Height of the input
    width,  # Width of the input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output data
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // width
    out_w = pid % width
    # Compute the input position
    input_h = out_h * stride - padding
    input_w = out_w * stride - padding
    # Compute the offset in the input tensor
    input_offset = (input_h * width + input_w) * in_channels
    # Load input values
    input_val = tl.load(input_ptr + input_offset, mask=(input_h >= 0) & (input_w >= 0) & (input_h < height) & (input_w < width), other=0.0)
    # Compute the output value
    output_val = input_val * tl.load(weight_ptr, mask=(input_h >= 0) & (input_w >= 0) & (input_h < height) & (input_w < width), other=0.0)
    # Store the output value
    output_offset = (out_h * width + out_w) * out_channels
    tl.store(output_ptr + output_offset, output_val)

@triton.jit
def conv3x3_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    stride,  # Stride of the convolution
    padding,  # Padding of the convolution
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    height,  # Height of the input
    width,  # Width of the input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output data
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // width
    out_w = pid % width
    # Compute the input position
    input_h = out_h * stride - padding
    input_w = out_w * stride - padding
    # Compute the offset in the input tensor
    input_offset = (input_h * width + input_w) * in_channels
    # Load input values
    input_val = tl.load(input_ptr + input_offset, mask=(input_h >= 0) & (input_w >= 0) & (input_h < height) & (input_w < width), other=0.0)
    # Compute the output value
    output_val = input_val * tl.load(weight_ptr, mask=(input_h >= 0) & (input_w >= 0) & (input_h < height) & (input_w < width), other=0.0)
    # Store the output value
    output_offset = (out_h * width + out_w) * out_channels
    tl.store(output_ptr + output_offset, output_val)

def triton_conv1x1(x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    out_channels = weight.shape[0]
    in_channels = weight.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    output = torch.empty((out_channels, height, width), device=x.device, dtype=x.dtype)
    num_elements = out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1x1_kernel[grid](x, weight, output, stride, padding, out_channels, in_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_conv3x3(x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    out_channels = weight.shape[0]
    in_channels = weight.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    output = torch.empty((out_channels, height, width), device=x.device, dtype=x.dtype)
    num_elements = out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3x3_kernel[grid](x, weight, output, stride, padding, out_channels, in_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)