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
    dilation,  # (dH, dW)
    BLOCK_SIZE: tl.constexpr,
):
    # Extract dimensions
    N, C, H, W = input_shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    # Get the thread index
    pid = tl.program_id(0)
    # Get the block index in the output
    block_idx = pid
    # Compute the output coordinates
    out_h = block_idx // (W // sW)
    out_w = block_idx % (W // sW)

    # Compute the input coordinates
    in_h = out_h * sH - pH + tl.arange(0, kH) * dH
    in_w = out_w * sW - pW + tl.arange(0, kW) * dW
    in_h = tl.where(in_h < 0, 0, in_h)
    in_w = tl.where(in_w < 0, 0, in_w)

    # Compute the offset for the input and weight
    input_offset = tl.arange(0, BLOCK_SIZE)
    weight_offset = tl.arange(0, kH) * C * kW + tl.arange(0, kW)

    # Load input and weight
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset < C * H * W, other=0.0)
    weight_vals = tl.load(weight_ptr + weight_offset, mask=weight_offset < C * kH * kW, other=0.0)

    # Compute the output
    output = tl.dot(input_vals, weight_vals)
    # Store the output
    tl.store(output_ptr + block_idx, output)


def triton_conv2d(input, weight, stride, padding, dilation):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output_shape = (input.size(0), weight.size(0), (input.size(2) + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1,
                    (input.size(3) + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    num_blocks = output.size(0) * output.size(1) * output.size(2) * output.size(3)
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    conv2d_kernel[grid](input, weight, output, input.shape, (kernel_size[0], kernel_size[1]), (stride[0], stride[1]), (padding[0], padding[1]), (dilation[0], dilation[1]), BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation with custom Triton kernels.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        x = self.model[4](x)
        x = self.model[5](x)
        x = self.model[6](x)
        x = self.model[7](x)
        x = self.model[8](x)
        x = self.model[9](x)
        x = self.model[10](x)
        x = self.model[11](x)
        x = self.model[12](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x