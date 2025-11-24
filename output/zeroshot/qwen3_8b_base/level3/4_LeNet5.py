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
    stride,  # Stride of the convolution
    kernel_size,  # Size of the kernel
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block ID
    block_id = pid // (GROUP_SIZE * GROUP_SIZE)
    # Get the group ID
    group_id = pid % (GROUP_SIZE * GROUP_SIZE)
    # Get the group coordinates
    group_row = group_id // GROUP_SIZE
    group_col = group_id % GROUP_SIZE

    # Compute the output position
    output_row = (block_id // GROUP_SIZE) * GROUP_SIZE + group_row
    output_col = (block_id % GROUP_SIZE) * GROUP_SIZE + group_col

    # Compute the input offset
    input_offset = (output_row * stride) * 32 + (output_col * stride)
    input_offset = input_offset * 32 + input_offset

    # Compute the weight offset
    weight_offset = group_row * kernel_size * kernel_size * NUM_GROUPS + group_col * kernel_size * kernel_size

    # Initialize the output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the block
    for i in range(BLOCK_SIZE):
        # Compute the input position
        input_row = output_row + i // 32
        input_col = output_col + i % 32
        input_pos = input_row * 32 * 32 + input_col * 32 + input_offset

        # Compute the weight position
        weight_row = i // 32
        weight_col = i % 32
        weight_pos = weight_offset + weight_row * kernel_size + weight_col

        # Load input and weight
        input_val = tl.load(input_ptr + input_pos, mask=tl.arange(0, 32) < 32, other=0.0)
        weight_val = tl.load(weight_ptr + weight_pos, mask=tl.arange(0, 32) < 32, other=0.0)

        # Multiply and accumulate
        output += input_val * weight_val

    # Store the output
    tl.store(output_ptr + output_row * 32 + output_col, output, mask=tl.arange(0, 32) < 32)


@triton.jit
def max_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    kernel_size,  # Size of the kernel
    stride,  # Stride of the pooling
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block ID
    block_id = pid // (BLOCK_SIZE * BLOCK_SIZE)
    # Get the block coordinates
    block_row = block_id // BLOCK_SIZE
    block_col = block_id % BLOCK_SIZE

    # Compute the output position
    output_row = block_row
    output_col = block_col

    # Compute the input offset
    input_offset = (output_row * stride) * 32 + (output_col * stride) * 32

    # Initialize the output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the block
    for i in range(BLOCK_SIZE):
        # Compute the input position
        input_row = output_row + i // 32
        input_col = output_col + i % 32
        input_pos = input_row * 32 + input_col + input_offset

        # Load input
        input_val = tl.load(input_ptr + input_pos, mask=tl.arange(0, 32) < 32, other=-float('inf'))

        # Update the output
        output[i] = tl.max(output[i], input_val)

    # Store the output
    tl.store(output_ptr + output_row * 32 + output_col, output, mask=tl.arange(0, 32) < 32)


@triton.jit
def linear_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in input
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block ID
    block_id = pid // BLOCK_SIZE
    # Get the offset within the block
    offset = pid % BLOCK_SIZE

    # Compute the input position
    input_pos = block_id * BLOCK_SIZE + offset
    input_val = tl.load(input_ptr + input_pos, mask=offset < n_elements, other=0.0)

    # Compute the weight position
    weight_pos = offset
    weight_val = tl.load(weight_ptr + weight_pos, mask=offset < n_elements, other=0.0)

    # Multiply and accumulate
    output_val = input_val * weight_val

    # Store the output
    tl.store(output_ptr + offset, output_val, mask=offset < n_elements)


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in input
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block ID
    block_id = pid // BLOCK_SIZE
    # Get the offset within the block
    offset = pid % BLOCK_SIZE

    # Compute the input position
    input_pos = block_id * BLOCK_SIZE + offset
    input_val = tl.load(input_ptr + input_pos, mask=offset < n_elements, other=0.0)

    # Apply ReLU
    output_val = tl.maximum(input_val, 0.0)

    # Store the output
    tl.store(output_ptr + offset, output_val, mask=offset < n_elements)


def triton_conv2d(input, weight, stride, kernel_size):
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the input
    n_elements = input.numel()

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, stride, kernel_size, BLOCK_SIZE=128, GROUP_SIZE=32, NUM_GROUPS=1)
    return output


def triton_max_pool2d(input, kernel_size, stride):
    # Ensure input is contiguous
    input = input.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the input
    n_elements = input.numel()

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    max_pool2d_kernel[grid](input, output, kernel_size, stride, BLOCK_SIZE=128)
    return output


def triton_linear(input, weight):
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the input
    n_elements = input.numel()

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    linear_kernel[grid](input, weight, output, n_elements, BLOCK_SIZE=128)
    return output


def triton_relu(input):
    # Ensure input is contiguous
    input = input.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the input
    n_elements = input.numel()

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # First convolutional layer with ReLU activation and max pooling
        x = triton_conv2d(x, self.conv1.weight, stride=1, kernel_size=5)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)

        # Second convolutional layer with ReLU activation and max pooling
        x = triton_conv2d(x, self.conv2.weight, stride=1, kernel_size=5)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)

        # First fully connected layer with ReLU activation
        x = triton_linear(x, self.fc1.weight)
        x = triton_relu(x)

        # Second fully connected layer with ReLU activation
        x = triton_linear(x, self.fc2.weight)
        x = triton_relu(x)

        # Final fully connected layer
        x = triton_linear(x, self.fc3.weight)

        return x