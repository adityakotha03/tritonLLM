import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_relu_kernel(
    a_ptr,  # Pointer to first input (matrix A)
    b_ptr,  # Pointer to second input (matrix B)
    out_ptr,  # Pointer to output (matrix C)
    m,  # Number of rows in A and C
    n,  # Number of columns in B and C
    k,  # Number of columns in A and rows in B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Each thread handles a single element of the output matrix
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    # Compute the offset for each element in A and B
    a_offsets = row[:, None] * k + col[None, :]
    b_offsets = col[None, :] * m + row[:, None]
    # Load A and B
    a = tl.load(a_ptr + a_offsets, mask=(row < m)[:, None] & (col < k)[None, :], other=0.0)
    b = tl.load(b_ptr + b_offsets, mask=(col < n)[None, :] & (row < m)[:, None], other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Apply ReLU
    c = tl.maximum(c, 0.0)
    # Store the result
    tl.store(out_ptr + row[:, None] * n + col[None, :], c, mask=(row < m)[:, None] & (col < n)[None, :])


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor, m: int, n: int, k: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    matmul_relu_kernel[ num_blocks ](a, b, out, m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    output_channels,  # Number of output channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    kernel_size,  # Size of kernel
    stride,  # Stride of the convolution
    padding,  # Padding added to both sides of the input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Each thread handles a single output element
    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the position in the output tensor
    out_row = out_idx // width
    out_col = out_idx % width
    # Compute the corresponding input region
    in_row = out_row * stride - padding
    in_col = out_col * stride - padding
    # Compute the input indices for each channel
    in_offsets = (in_row[None, None, None, :] * width + in_col[None, None, :, None]) * input_channels + tl.arange(0, input_channels)
    # Compute the weight indices
    weight_offsets = (tl.arange(0, output_channels)[:, None, None] * input_channels + tl.arange(0, input_channels)[None, :, None]) * kernel_size * kernel_size + tl.arange(0, kernel_size)[None, None, :] * kernel_size + tl.arange(0, kernel_size)[None, :, None]
    # Load input and weight
    input_vals = tl.load(input_ptr + in_offsets, mask=(in_row < height)[:, None, None] & (in_col < width)[None, None, :], other=0.0)
    weight_vals = tl.load(weight_ptr + weight_offsets, mask=(tl.arange(0, output_channels)[:, None, None] < output_channels) & (tl.arange(0, input_channels)[None, :, None] < input_channels) & (tl.arange(0, kernel_size)[None, None, :] < kernel_size) & (tl.arange(0, kernel_size)[None, :, None] < kernel_size), other=0.0)
    # Compute the convolution
    c = tl.sum(input_vals * weight_vals, axis=(1, 2, 3))
    # Store the result
    tl.store(output_ptr + out_idx, c, mask=(out_row < height) & (out_col < width))


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, input_channels: int, output_channels: int, height: int, width: int, kernel_size: int, stride: int, padding: int):
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
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((output_height, output_width, output_channels), dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    num_blocks = (output_height * output_width + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    conv2d_kernel[ num_blocks ](input, weight, output, input_channels, output_channels, height, width, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized MobileNetV2 architecture with custom Triton kernels for matmul and conv2d operations.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                # Replace the default conv2d with custom Triton kernel
                conv2d, use_res_connect = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                if not use_res_connect:
                    # Replace the conv2d with the Triton kernel
                    conv2d = nn.Conv2d(input_channel, output_channel, 3, stride, 1, groups=input_channel, bias=False)
                    conv2d.weight = torch.nn.Parameter(triton_conv2d(
                        conv2d.weight, 
                        conv2d.weight, 
                        input_channels=input_channel, 
                        output_channels=output_channel, 
                        height=1, 
                        width=1, 
                        kernel_size=3, 
                        stride=stride, 
                        padding=1
                    ).data)
                features.append(conv2d)
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))

        # Final layer
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x