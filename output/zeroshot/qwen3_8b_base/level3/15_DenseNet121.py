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
    stride_h,  # Stride in height dimension
    stride_w,  # Stride in width dimension
    kernel_h,  # Height of the kernel
    kernel_w,  # Width of the kernel
    input_channels,  # Number of input channels
    output_channels,  # Number of output channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the block's position in the output
    block_h = pid // (width // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)
    # Compute the starting position in the input
    offset_h = block_h * stride_h
    offset_w = block_w * stride_w
    # Compute the starting position in the output
    out_h = block_h
    out_w = block_w
    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input position
            input_h = offset_h + i
            input_w = offset_w + j
            # Check if input is within bounds
            if input_h >= height or input_w >= width:
                continue
            # Compute the output position
            out_h = block_h
            out_w = block_w
            # Load the input value
            input_val = tl.load(input_ptr + (input_h * width + input_w))
            # Multiply by weight and accumulate
            output_val = tl.load(output_ptr + (out_h * width + out_w))
            output_val += input_val * tl.load(weight_ptr + (input_h * kernel_w + input_w))
            tl.store(output_ptr + (out_h * width + out_w), output_val)

def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride_h: int, stride_w: int):
    """
    This function wraps the Triton kernel call for 2D convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)
    # Grid size
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    # Launch the kernel
    conv2d_kernel[grid](input, weight, output, stride_h, stride_w, 3, 3, input.size(1), output.size(1), input.size(2), input.size(3), BLOCK_SIZE=128)
    return output


@triton.jit
def batchnorm_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    mean_ptr,  # Pointer to mean tensor
    rstd_ptr,  # Pointer to reciprocal standard deviation tensor
    output_ptr,  # Pointer to output tensor
    num_channels,  # Number of channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the block's position in the output
    block_h = pid // (width // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)
    # Compute the starting position in the input
    offset_h = block_h * 1
    offset_w = block_w * 1
    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input position
            input_h = offset_h + i
            input_w = offset_w + j
            # Check if input is within bounds
            if input_h >= height or input_w >= width:
                continue
            # Compute the output position
            out_h = block_h
            out_w = block_w
            # Load the input value
            input_val = tl.load(input_ptr + (input_h * width + input_w))
            # Compute the normalized value
            normalized_val = (input_val - tl.load(mean_ptr)) * tl.load(rstd_ptr)
            # Apply weight and bias
            output_val = normalized_val * tl.load(weight_ptr) + tl.load(bias_ptr)
            tl.store(output_ptr + (out_h * width + out_w), output_val)

def triton_batchnorm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor):
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
    # Grid size
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    # Launch the kernel
    batchnorm_kernel[grid](input, weight, bias, mean, rstd, output, input.size(1), input.size(2), input.size(3), BLOCK_SIZE=128)
    return output


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    num_channels,  # Number of channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the block's position in the output
    block_h = pid // (width // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)
    # Compute the starting position in the input
    offset_h = block_h * 1
    offset_w = block_w * 1
    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input position
            input_h = offset_h + i
            input_w = offset_w + j
            # Check if input is within bounds
            if input_h >= height or input_w >= width:
                continue
            # Compute the output position
            out_h = block_h
            out_w = block_w
            # Load the input value
            input_val = tl.load(input_ptr + (input_h * width + input_w))
            # Apply ReLU
            output_val = tl.maximum(input_val, 0.0)
            tl.store(output_ptr + (out_h * width + out_w), output_val)

def triton_relu(input: torch.Tensor):
    """
    This function wraps the Triton kernel call for ReLU.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)
    # Grid size
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    # Launch the kernel
    relu_kernel[grid](input, output, input.size(1), input.size(2), input.size(3), BLOCK_SIZE=128)
    return output


@triton.jit
def dropout_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    p,  # Dropout probability
    seed,  # Random seed
    num_channels,  # Number of channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the block's position in the output
    block_h = pid // (width // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)
    # Compute the starting position in the input
    offset_h = block_h * 1
    offset_w = block_w * 1
    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input position
            input_h = offset_h + i
            input_w = offset_w + j
            # Check if input is within bounds
            if input_h >= height or input_w >= width:
                continue
            # Compute the output position
            out_h = block_h
            out_w = block_w
            # Load the input value
            input_val = tl.load(input_ptr + (input_h * width + input_w))
            # Apply dropout
            if tl.random(seed) > p:
                output_val = 0.0
            else:
                output_val = input_val / (1.0 - p)
            tl.store(output_ptr + (out_h * width + out_w), output_val)

def triton_dropout(input: torch.Tensor, p: float, seed: int):
    """
    This function wraps the Triton kernel call for dropout.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)
    # Grid size
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    # Launch the kernel
    dropout_kernel[grid](input, output, p, seed, input.size(1), input.size(2), input.size(3), BLOCK_SIZE=128)
    return output


@triton.jit
def avg_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    kernel_size,  # Size of the kernel
    stride,  # Stride of the pooling
    input_channels,  # Number of input channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the block's position in the output
    block_h = pid // (width // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)
    # Compute the starting position in the input
    offset_h = block_h * stride
    offset_w = block_w * stride
    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input position
            input_h = offset_h + i
            input_w = offset_w + j
            # Check if input is within bounds
            if input_h >= height or input_w >= width:
                continue
            # Compute the output position
            out_h = block_h
            out_w = block_w
            # Load the input value
            input_val = tl.load(input_ptr + (input_h * width + input_w))
            # Accumulate the sum
            output_val = tl.load(output_ptr + (out_h * width + out_w))
            output_val += input_val
            tl.store(output_ptr + (out_h * width + out_w), output_val)

def triton_avg_pool2d(input: torch.Tensor, kernel_size: int, stride: int):
    """
    This function wraps the Triton kernel call for average pooling.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty((input.size(0), input.size(1), input.size(2) // stride, input.size(3) // stride), device=input.device)
    # Grid size
    grid = lambda meta: (meta["BLOCK_SIZE"],)
    # Launch the kernel
    avg_pool2d_kernel[grid](input, output, kernel_size, stride, input.size(1), input.size(2), input.size(3), BLOCK_SIZE=128)
    # Normalize the output
    output /= kernel_size * kernel_size
    return output


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_features = 64
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            triton.BatchNorm2d(in_features),
            triton.ReLU(),
            triton.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            triton.Dropout(0.0)
        )

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            triton.BatchNorm2d(num_input_features),
            triton.ReLU(),
            triton.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            triton.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)