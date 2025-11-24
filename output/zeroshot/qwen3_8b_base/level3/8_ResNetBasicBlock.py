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
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Extract the block dimensions
    N, C, H, W = input_shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    # Compute the output shape
    out_H = (H + 2 * pH - kH) // sH + 1
    out_W = (W + 2 * pW - kW) // sW + 1

    # Compute the offset for the current thread
    pid = tl.program_id(0)
    block_id = pid // (out_H * out_W)
    block_h = (pid // out_W) % out_H
    block_w = pid % out_W

    # Compute the offset in the output tensor
    out_offset = block_id * out_H * out_W * out_channels
    out_offset += block_h * out_W * out_channels
    out_offset += block_w * out_channels

    # Compute the input offset for the current block
    input_offset = block_id * (H * W * in_channels)
    input_offset += block_h * W * in_channels
    input_offset += block_w * in_channels

    # Compute the weight offset
    weight_offset = 0
    for g in range(GROUP_SIZE):
        weight_offset += (out_channels // GROUP_SIZE) * in_channels * kH * kW

    # Load the weights
    weights = tl.load(weight_ptr + weight_offset, shape=(out_channels // GROUP_SIZE, in_channels, kH, kW), mask=tl.full((out_channels // GROUP_SIZE, in_channels, kH, kW), True, dtype=tl.int32))

    # Compute the input and output indices
    input_idx = tl.arange(0, BLOCK_SIZE)
    input_idx = input_idx + input_offset
    input_idx = input_idx + tl.arange(0, in_channels) * H * W
    input_idx = input_idx + tl.arange(0, kH) * W
    input_idx = input_idx + tl.arange(0, kW)

    # Compute the output indices
    output_idx = tl.arange(0, out_channels // GROUP_SIZE)
    output_idx = output_idx + out_offset
    output_idx = output_idx + tl.arange(0, in_channels) * kH * kW
    output_idx = output_idx + tl.arange(0, kH) * kW
    output_idx = output_idx + tl.arange(0, kW)

    # Compute the input and output values
    input_vals = tl.load(input_ptr + input_idx, mask=tl.full((BLOCK_SIZE, in_channels, kH, kW), True, dtype=tl.int32), other=0.0)
    output_vals = tl.load(output_ptr + output_idx, mask=tl.full((out_channels // GROUP_SIZE, in_channels, kH, kW), True, dtype=tl.int32), other=0.0)

    # Compute the convolution
    output_vals = tl.dot(input_vals, weights)

    # Store the output
    tl.store(output_ptr + output_idx, output_vals, mask=tl.full((out_channels // GROUP_SIZE, in_channels, kH, kW), True, dtype=tl.int32))


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride, padding):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    N, C, H, W = input.shape
    kH, kW = weight.shape[2], weight.shape[3]
    sH, sW = stride
    pH, pW = padding
    out_H = (H + 2 * pH - kH) // sH + 1
    out_W = (W + 2 * pW - kW) // sW + 1
    out_channels = weight.shape[0]
    out = torch.zeros((N, out_channels, out_H, out_W), device=input.device)

    # Determine the number of blocks needed
    num_blocks = (N * out_H * out_W * out_channels) // (BLOCK_SIZE * GROUP_SIZE)
    grid = lambda meta: (num_blocks,)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, out, (N, C, H, W), (kH, kW), (sH, sW), (pH, pW), out_channels, C, BLOCK_SIZE=128, GROUP_SIZE=1)
    return out


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out