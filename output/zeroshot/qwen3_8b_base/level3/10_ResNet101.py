import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
    GROUP_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get block index in the output
    block_idx = pid // (input_shape[0] * input_shape[1])
    # Get output channel index
    oc = pid % (input_shape[0] * input_shape[1])
    # Get output channel index
    oc = oc % input_shape[0]
    # Get input channel index
    ic = pid % input_shape[1]
    # Get output position
    oh = (block_idx // input_shape[2]) % input_shape[3]
    ow = block_idx % input_shape[2]

    # Compute output offset
    out_offset = oc * input_shape[2] * input_shape[3] + oh * input_shape[3] + ow
    # Compute input offset
    in_offset = ic * input_shape[2] * input_shape[3] + oh * input_shape[3] + ow

    # Load input data
    input_data = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < input_shape[3], other=0.0)
    # Load weight data
    weight_data = tl.load(weight_ptr + oc * input_shape[1] * kernel_size[0] * kernel_size[1] + ic * kernel_size[0] * kernel_size[1] + tl.arange(0, kernel_size[0]) * kernel_size[1] + tl.arange(0, kernel_size[1]), mask=tl.arange(0, kernel_size[0]) < kernel_size[0] and tl.arange(0, kernel_size[1]) < kernel_size[1], other=0.0)
    # Compute convolution
    output = tl.dot(input_data, weight_data)
    # Store output
    tl.store(output_ptr + out_offset, output)

def triton_conv2d(input, weight, bias, stride, padding, dilation):
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)

    # Compute output shape
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1

    # Define block size
    BLOCK_SIZE = 128
    GROUP_SIZE = 128

    # Launch kernel
    grid = lambda meta: (batch_size * out_channels * out_height * out_width,)
    conv2d_kernel[grid](input, weight, output, (batch_size, in_channels, out_channels, out_height, out_width), (kernel_height, kernel_width), (stride[0], stride[1]), (padding[0], padding[1]), (dilation[0], dilation[1]), BLOCK_SIZE, GROUP_SIZE)

    # Add bias
    output += bias

    return output

@triton.jit
def batch_norm_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get output position
    oh = (pid // input_shape[2]) % input_shape[3]
    ow = pid % input_shape[2]

    # Compute input offset
    in_offset = oh * input_shape[3] + ow
    # Compute output offset
    out_offset = oh * input_shape[3] + ow

    # Load input data
    input_data = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < input_shape[3], other=0.0)
    # Load weight data
    weight_data = tl.load(weight_ptr, other=1.0)
    # Load bias data
    bias_data = tl.load(bias_ptr, other=0.0)
    # Load mean and variance
    mean = tl.load(mean_ptr, other=0.0)
    var = tl.load(var_ptr, other=1.0)

    # Compute batch normalization
    output = (input_data - mean) * tl.rsqrt(var + eps) * weight_data + bias_data
    # Store output
    tl.store(output_ptr + out_offset, output)

def triton_batch_norm(input, weight, bias, mean, var, eps=1e-5):
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and mean.is_cuda and var.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    output = torch.empty_like(input)

    # Compute output shape
    batch_size, in_channels, in_height, in_width = input.shape

    # Define block size
    BLOCK_SIZE = 128

    # Launch kernel
    grid = lambda meta: (batch_size * in_channels * in_height * in_width,)
    batch_norm_kernel[grid](input, weight, bias, mean, var, output, (batch_size, in_channels, in_height, in_width), eps, BLOCK_SIZE)

    return output

@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get output position
    oh = (pid // input_shape[2]) % input_shape[3]
    ow = pid % input_shape[2]

    # Compute input offset
    in_offset = oh * input_shape[3] + ow
    # Compute output offset
    out_offset = oh * input_shape[3] + ow

    # Load input data
    input_data = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < input_shape[3], other=0.0)
    # Compute ReLU
    output = tl.maximum(input_data, 0.0)
    # Store output
    tl.store(output_ptr + out_offset, output)

def triton_relu(input):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    # Compute output shape
    batch_size, in_channels, in_height, in_width = input.shape

    # Define block size
    BLOCK_SIZE = 128

    # Launch kernel
    grid = lambda meta: (batch_size * in_channels * in_height * in_width,)
    relu_kernel[grid](input, output, (batch_size, in_channels, in_height, in_width), BLOCK_SIZE)

    return output

@triton.jit
def avg_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get output position
    oh = (pid // input_shape[2]) % input_shape[3]
    ow = pid % input_shape[2]

    # Compute input offset
    in_offset = oh * input_shape[3] + ow
    # Compute output offset
    out_offset = oh * input_shape[3] + ow

    # Load input data
    input_data = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < input_shape[3], other=0.0)
    # Compute average pooling
    output = tl.sum(input_data) / (kernel_size[0] * kernel_size[1])
    # Store output
    tl.store(output_ptr + out_offset, output)

def triton_avg_pool2d(input, kernel_size, stride, padding):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty((input.shape[0], input.shape[1], input.shape[2] // stride[0], input.shape[3] // stride[1]), device=input.device)

    # Compute output shape
    batch_size, in_channels, in_height, in_width = input.shape
    out_height = (in_height + 2 * padding[0] - kernel_size[0] - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_size[1] - 1) // stride[1] + 1

    # Define block size
    BLOCK_SIZE = 128

    # Launch kernel
    grid = lambda meta: (batch_size * in_channels * out_height * out_width,)
    avg_pool2d_kernel[grid](input, output, (batch_size, in_channels, in_height, in_width), (kernel_size[0], kernel_size[1]), (stride[0], stride[1]), (padding[0], padding[1]), BLOCK_SIZE)

    return output

@triton.jit
def linear_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C)
    output_shape,  # (N, D)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get output position
    n = pid // input_shape[1]
    d = pid % input_shape[1]

    # Compute input offset
    in_offset = n * input_shape[1] + d
    # Compute output offset
    out_offset = n * output_shape[1] + d

    # Load input data
    input_data = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < input_shape[1], other=0.0)
    # Load weight data
    weight_data = tl.load(weight_ptr + d * output_shape[1] + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < output_shape[1], other=0.0)
    # Compute linear transformation
    output = tl.dot(input_data, weight_data)
    # Store output
    tl.store(output_ptr + out_offset, output)

def triton_linear(input, weight, bias, eps=1e-5):
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output = torch.empty((input.shape[0], weight.shape[1]), device=input.device)

    # Compute output shape
    batch_size, in_features = input.shape
    out_features = weight.shape[1]

    # Define block size
    BLOCK_SIZE = 128

    # Launch kernel
    grid = lambda meta: (batch_size * out_features,)
    linear_kernel[grid](input, weight, output, (batch_size, in_features), (batch_size, out_features), BLOCK_SIZE)

    # Add bias
    output += bias

    return output

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_custom(self, x):
        # Custom forward with Triton kernels
        x = triton_conv2d(x, self.conv1.weight, self.conv1.bias, (2, 2), (3, 3), (1, 1))
        x = triton_batch_norm(x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var)
        x = triton_relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = triton_linear(x, self.fc.weight, self.fc.bias)

        return x