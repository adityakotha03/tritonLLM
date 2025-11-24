import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_channels, output_channels, kernel_size, stride, padding,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute the position within the block
    pid = tl.program_id(axis=0)
    # Compute the block's starting position
    block_m = pid // (BLOCK_N * BLOCK_K)
    block_n = (pid // BLOCK_K) % BLOCK_N
    block_k = pid % BLOCK_K

    # Compute the offset for the block
    m = block_m * BLOCK_M
    n = block_n * BLOCK_N
    k = block_k * BLOCK_K

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the kernel dimensions
    for k_idx in range(BLOCK_K):
        # Load input and weight
        input_block = tl.load(input_ptr + (m + k_idx) * input_channels + tl.arange(0, BLOCK_M) * input_channels + tl.arange(0, BLOCK_K) * input_channels, mask=(m + k_idx) < input_channels, other=0.0)
        weight_block = tl.load(weight_ptr + (k_idx) * output_channels + tl.arange(0, BLOCK_K) * output_channels + tl.arange(0, BLOCK_N) * output_channels, mask=(k_idx) < output_channels, other=0.0)
        # Perform matrix multiplication
        acc += tl.dot(input_block, weight_block)
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, output_channels), other=0.0)
        acc += bias[None, :]

    # Store the result
    tl.store(output_ptr + n * output_channels + tl.arange(0, BLOCK_N) * output_channels + tl.arange(0, BLOCK_M), acc)

def triton_conv2d(input, weight, bias, stride, padding):
    # Ensure the inputs are on the GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Output tensor
    output = torch.empty((input.size(0), weight.size(0), input.size(2) // stride, input.size(3) // stride), dtype=input.dtype, device=input.device)

    # Compute the grid size
    num_blocks = (input.size(2) // stride) * (input.size(3) // stride)
    grid = (num_blocks,)

    # Launch the kernel
    conv2d_kernel[grid](
        input.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
        input.size(1), weight.size(0), weight.size(2), stride, padding,
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=16
    )
    return output

@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, out_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position within the block
    pid = tl.program_id(axis=0)
    # Compute the block's starting position
    row = pid // (BLOCK_SIZE)
    col = pid % (BLOCK_SIZE)

    # Compute the offset for the block
    row_offsets = row * BLOCK_SIZE
    col_offsets = col * BLOCK_SIZE

    # Load matrix A and B
    a = tl.load(a_ptr + row_offsets + tl.arange(0, BLOCK_SIZE) * k, mask=(row_offsets + tl.arange(0, BLOCK_SIZE)) < m, other=0.0)
    b = tl.load(b_ptr + col_offsets + tl.arange(0, BLOCK_SIZE) * n, mask=(col_offsets + tl.arange(0, BLOCK_SIZE)) < n, other=0.0)

    # Compute the matrix multiplication
    c = tl.dot(a, b)

    # Apply ReLU
    c = tl.maximum(c, 0.0)

    # Store the result
    tl.store(out_ptr + row_offsets + tl.arange(0, BLOCK_SIZE) * n + col_offsets, c, mask=(row_offsets + tl.arange(0, BLOCK_SIZE)) < m)

def triton_matmul_relu(a, b):
    # Ensure the inputs are on the GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Output tensor
    output = torch.empty((a.size(0), b.size(1)), dtype=a.dtype, device=a.device)

    # Compute the grid size
    num_blocks = (a.size(0) + 128 - 1) // 128
    grid = (num_blocks,)

    # Launch the kernel
    matmul_relu_kernel[grid](
        a.data_ptr(), b.data_ptr(), output.data_ptr(),
        a.size(0), b.size(1), a.size(1),
        BLOCK_SIZE=128
    )
    return output

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position within the block
    pid = tl.program_id(axis=0)
    # Compute the block's starting position
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    input = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    # Compute max value in the block
    max_val = tl.max(input, axis=0)
    # Subtract max value to avoid overflow
    input -= max_val
    # Compute exponentials
    exp_input = tl.exp(input)
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_input, axis=0)
    # Compute softmax
    softmax = exp_input / sum_exp
    # Store the result
    tl.store(output_ptr + offsets, softmax, mask=mask)

def triton_softmax(input):
    # Ensure the input is on the GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Output tensor
    output = torch.empty_like(input)

    # Compute the grid size
    num_blocks = (input.size(0) * input.size(1) + 128 - 1) // 128
    grid = (num_blocks,)

    # Launch the kernel
    softmax_kernel[grid](
        input.data_ptr(), output.data_ptr(),
        input.numel(),
        BLOCK_SIZE=128
    )
    return output

def triton_avgpool(input, kernel_size, stride, padding):
    # Ensure the input is on the GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Compute output dimensions
    output_size = (input.size(2) + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((input.size(0), input.size(1), output_size, output_size), dtype=input.dtype, device=input.device)

    # Compute the grid size
    num_blocks = (input.size(2) + 128 - 1) // 128
    grid = (num_blocks,)

    # Launch the kernel
    avgpool_kernel[grid](
        input.data_ptr(), output.data_ptr(),
        input.size(0), input.size(1), input.size(2), input.size(3),
        kernel_size, stride, padding,
        BLOCK_SIZE=128
    )
    return output

@triton.jit
def avgpool_kernel(
    input_ptr, output_ptr,
    batch, channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position within the block
    pid = tl.program_id(axis=0)
    # Compute the block's starting position
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < height
    # Load input values
    input = tl.load(input_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    # Compute average
    avg = tl.sum(input) / BLOCK_SIZE
    # Store the result
    tl.store(output_ptr + offsets, avg, mask=mask)

def triton_linear(input, weight, bias):
    # Ensure the inputs are on the GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output tensor
    output = torch.empty((input.size(0), weight.size(0)), dtype=input.dtype, device=input.device)

    # Compute the grid size
    num_blocks = (input.size(0) + 128 - 1) // 128
    grid = (num_blocks,)

    # Launch the kernel
    linear_kernel[grid](
        input.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
        input.size(0), input.size(1), weight.size(0),
        BLOCK_SIZE=128
    )
    return output

@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position within the block
    pid = tl.program_id(axis=0)
    # Compute the block's starting position
    row = pid // (BLOCK_SIZE)
    col = pid % (BLOCK_SIZE)

    # Compute the offset for the block
    row_offsets = row * BLOCK_SIZE
    col_offsets = col * BLOCK_SIZE

    # Load matrix A and B
    a = tl.load(input_ptr + row_offsets + tl.arange(0, BLOCK_SIZE) * n, mask=(row_offsets + tl.arange(0, BLOCK_SIZE)) < m, other=0.0)
    b = tl.load(weight_ptr + col_offsets + tl.arange(0, BLOCK_SIZE) * k, mask=(col_offsets + tl.arange(0, BLOCK_SIZE)) < k, other=0.0)

    # Compute the matrix multiplication
    c = tl.dot(a, b)

    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, k), other=0.0)
        c += bias[None, :]

    # Store the result
    tl.store(output_ptr + row_offsets + tl.arange(0, BLOCK_SIZE) * k, c, mask=(row_offsets + tl.arange(0, BLOCK_SIZE)) < m)