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
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get thread ID within the block
    thread_id = tl.program_id(1)
    # Compute the block's position in the output
    block_h = pid // (GROUP_SIZE * BLOCK_W)
    block_w = (pid // GROUP_SIZE) % BLOCK_W
    block_group = pid % GROUP_SIZE

    # Compute the offset in the output
    offset_h = block_h * BLOCK_H
    offset_w = block_w * BLOCK_W
    offset_group = block_group * BLOCK_H * BLOCK_W

    # Compute the input offset
    input_offset = offset_h * input_shape[2] + offset_w * input_shape[3]
    input_offset += padding[0] * input_shape[2] + padding[1] * input_shape[3]
    input_offset += input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

    # Compute the weight offset
    weight_offset = block_group * input_shape[1] + offset_h * input_shape[1] + offset_w

    # Compute the output offset
    output_offset = offset_h * input_shape[2] + offset_w * input_shape[3]
    output_offset += input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

    # Iterate over the block
    for i in range(BLOCK_H):
        for j in range(BLOCK_W):
            # Compute the input index
            input_idx = input_offset + i * input_shape[2] + j
            # Compute the weight index
            weight_idx = weight_offset + i * input_shape[1] + j
            # Load input and weight
            input_val = tl.load(input_ptr + input_idx, mask=(input_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]), other=0.0)
            weight_val = tl.load(weight_ptr + weight_idx, mask=(weight_idx < input_shape[1] * kernel_size[0] * kernel_size[1]), other=0.0)
            # Multiply and accumulate
            output_val = tl.dot(input_val, weight_val)
            # Store output
            tl.store(output_ptr + output_offset, output_val, mask=(output_offset < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]))

    return


def triton_conv2d(input, weight, bias, stride, padding, kernel_size):
    """
    This function wraps the Triton kernel call for convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)

    # Compute output shape
    output_shape = [input.shape[0], weight.shape[0], (input.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1, (input.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1]

    # Determine block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    GROUP_SIZE = 1

    # Determine grid and block sizes
    num_blocks = (output_shape[2] * output_shape[3]) // (BLOCK_H * BLOCK_W) + 1
    num_threads_per_block = BLOCK_H * BLOCK_W * GROUP_SIZE

    # Launch the Triton kernel
    conv2d_kernel[ (num_blocks, num_threads_per_block) ](input, weight, output, input.shape, kernel_size, stride, padding, BLOCK_H, BLOCK_W, GROUP_SIZE)
    return output


@triton.jit
def matmul_relu_kernel(
    a_ptr,  # Pointer to first input
    b_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block's position in the output
    block_row = pid // (BLOCK_SIZE // 2)
    block_col = pid % (BLOCK_SIZE // 2)

    # Compute the offset in the output
    offset_row = block_row * BLOCK_SIZE
    offset_col = block_col * BLOCK_SIZE

    # Compute the input offsets
    a_offset = offset_row * K
    b_offset = offset_col * K

    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the row and column indices
            row = offset_row + i
            col = offset_col + j
            # Compute the value
            val = 0.0
            for k in range(K):
                a = tl.load(a_ptr + a_offset + k, mask=(a_offset + k < M * K), other=0.0)
                b = tl.load(b_ptr + b_offset + k, mask=(b_offset + k < K * N), other=0.0)
                val += a * b
            # Apply ReLU
            val = tl.maximum(val, 0.0)
            # Store the result
            tl.store(out_ptr + row * N + col, val, mask=(row < M) and (col < N))

    return


def triton_matmul_relu(a, b):
    """
    This function wraps the Triton kernel call for matrix multiplication with ReLU.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

    # Determine block size
    BLOCK_SIZE = 128

    # Determine grid and block sizes
    num_blocks = (a.shape[0] * b.shape[1]) // (BLOCK_SIZE * BLOCK_SIZE) + 1
    num_threads_per_block = BLOCK_SIZE * BLOCK_SIZE

    # Launch the Triton kernel
    matmul_relu_kernel[ (num_blocks, num_threads_per_block) ](a, b, out, a.shape[0], b.shape[1], a.shape[1], BLOCK_SIZE)
    return out


@triton.jit
def linear_relu_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C)
    weight_shape,  # (C, D)
    bias_shape,  # (D,)
    output_shape,  # (N, D)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get thread ID within the block
    thread_id = tl.program_id(1)
    # Compute the block's position in the output
    block_n = pid // (BLOCK_SIZE)
    block_d = pid % (BLOCK_SIZE)

    # Compute the offset in the output
    offset_n = block_n * BLOCK_SIZE
    offset_d = block_d * BLOCK_SIZE

    # Compute the input offset
    input_offset = offset_n * input_shape[1] + offset_d
    input_offset += input_shape[0] * input_shape[1]

    # Compute the weight offset
    weight_offset = offset_d * weight_shape[0]

    # Compute the bias offset
    bias_offset = offset_d

    # Compute the output offset
    output_offset = offset_n * output_shape[1] + offset_d
    output_offset += output_shape[0] * output_shape[1]

    # Iterate over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input index
            input_idx = input_offset + i
            # Compute the weight index
            weight_idx = weight_offset + j
            # Load input and weight
            input_val = tl.load(input_ptr + input_idx, mask=(input_idx < input_shape[0] * input_shape[1]), other=0.0)
            weight_val = tl.load(weight_ptr + weight_idx, mask=(weight_idx < weight_shape[0] * weight_shape[1]), other=0.0)
            # Multiply and accumulate
            output_val = input_val * weight_val
            # Add bias
            output_val += tl.load(bias_ptr + bias_offset, mask=(bias_offset < bias_shape[0]), other=0.0)
            # Apply ReLU
            output_val = tl.maximum(output_val, 0.0)
            # Store output
            tl.store(output_ptr + output_offset, output_val, mask=(output_offset < output_shape[0] * output_shape[1]))

    return


def triton_linear_relu(input, weight, bias):
    """
    This function wraps the Triton kernel call for linear layer with ReLU.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output = torch.empty((input.shape[0], weight.shape[1]), device=input.device, dtype=input.dtype)

    # Determine block size
    BLOCK_SIZE = 128

    # Determine grid and block sizes
    num_blocks = (input.shape[0] * weight.shape[1]) // (BLOCK_SIZE * BLOCK_SIZE) + 1
    num_threads_per_block = BLOCK_SIZE * BLOCK_SIZE

    # Launch the Triton kernel
    linear_relu_kernel[ (num_blocks, num_threads_per_block) ](input, weight, bias, output, input.shape, weight.shape, bias.shape, output.shape, BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x