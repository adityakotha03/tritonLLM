import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in, H, W)
    weight_shape,  # (C_out, C_in, K, K)
    output_shape,  # (N, C_out, H_out, W_out)
    stride,  # Stride for convolution
    padding,  # Padding for convolution
    dilation,  # Dilation for convolution
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Compute the 2D index in the output
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    pid3 = tl.program_id(2)
    pid4 = tl.program_id(3)

    # Compute the output index
    out_h = pid
    out_w = pid2
    out_c = pid3
    out_n = pid4

    # Compute the input indices
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Compute the output shape
    N, C_out, H_out, W_out = output_shape
    C_in, K, _ = weight_shape

    # Compute the input shape
    _, C_in, H_in, W_in = input_shape

    # Compute the output index
    out_idx = out_n * C_out * H_out * W_out + out_c * H_out * W_out + out_h * W_out + out_w
    out_idx = tl.load(output_ptr + out_idx, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute the weight indices
    weight_idx = out_c * C_in * K * K + tl.arange(0, C_in) * K * K + tl.arange(0, K) * K + tl.arange(0, K)
    weight_idx = tl.load(weight_ptr + weight_idx, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute the input indices
    input_idx = out_n * C_in * H_in * W_in + tl.arange(0, C_in) * H_in * W_in + in_h * W_in + tl.arange(0, W_in)
    input_idx = tl.load(input_ptr + input_idx, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute the convolution
    out = tl.sum(input_idx * weight_idx, axis=0)
    out += tl.load(bias_ptr + out_c, mask=tl.arange(0, 1) < 1, other=0.0)

    # Store the result
    tl.store(output_ptr + out_idx, out, mask=tl.arange(0, 1) < 1)


@triton.jit
def linear_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in)
    weight_shape,  # (C_out, C_in)
    output_shape,  # (N, C_out)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 1D index in the output
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)

    # Compute the output index
    out_c = pid
    out_n = pid2

    # Compute the input index
    in_c = tl.arange(0, input_shape[1])
    in_n = out_n
    input_idx = in_n * input_shape[1] + in_c
    input_idx = tl.load(input_ptr + input_idx, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute the weight index
    weight_idx = out_c * input_shape[1] + in_c
    weight_idx = tl.load(weight_ptr + weight_idx, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute the linear operation
    out = tl.sum(input_idx * weight_idx, axis=0)
    out += tl.load(bias_ptr + out_c, mask=tl.arange(0, 1) < 1, other=0.0)

    # Store the result
    tl.store(output_ptr + out_n * output_shape[1] + out_c, out, mask=tl.arange(0, 1) < 1)


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 4D index in the output
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    pid3 = tl.program_id(2)
    pid4 = tl.program_id(3)

    # Compute the output index
    out_n = pid
    out_c = pid2
    out_h = pid3
    out_w = pid4

    # Compute the input index
    input_idx = out_n * input_shape[1] * input_shape[2] * input_shape[3] + out_c * input_shape[2] * input_shape[3] + out_h * input_shape[3] + out_w
    input_val = tl.load(input_ptr + input_idx, mask=tl.arange(0, 1) < 1, other=0.0)
    output_val = tl.maximum(input_val, 0.0)
    tl.store(output_ptr + input_idx, output_val, mask=tl.arange(0, 1) < 1)


@triton.jit
def max_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    output_shape,  # (N, C, H_out, W_out)
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 4D index in the output
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    pid3 = tl.program_id(2)
    pid4 = tl.program_id(3)

    # Compute the output index
    out_n = pid
    out_c = pid2
    out_h = pid3
    out_w = pid4

    # Compute the input indices
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    in_h_end = in_h_start + kernel_size
    in_w_end = in_w_start + kernel_size

    # Compute the input index
    input_idx = out_n * input_shape[1] * input_shape[2] * input_shape[3] + out_c * input_shape[2] * input_shape[3] + in_h_start * input_shape[3] + in_w_start
    input_val = tl.load(input_ptr + input_idx, mask=tl.arange(0, 1) < 1, other=-float('inf'))
    max_val = tl.max(input_val, axis=0)
    tl.store(output_ptr + out_n * output_shape[1] * output_shape[2] * output_shape[3] + out_c * output_shape[2] * output_shape[3] + out_h * output_shape[3] + out_w, max_val, mask=tl.arange(0, 1) < 1)


def triton_conv2d(input, weight, bias, stride, padding, dilation, output_shape):
    input_shape = input.shape
    weight_shape = weight.shape
    output_shape = output_shape

    # Ensure input, weight, bias, and output are on the GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and output.is_cuda, "Tensors must be on CUDA."

    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output = output.contiguous()

    # Determine the number of blocks needed
    N, C_in, H_in, W_in = input_shape
    C_out, _, K, _ = weight_shape
    H_out = (H_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    # Calculate the number of blocks
    num_blocks = (N * C_out * H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks2 = (C_in * K * K + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks3 = (N * H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks4 = (C_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, bias, output, input_shape, weight_shape, output_shape, stride, padding, dilation, BLOCK_SIZE=128, GROUP_SIZE=1)
    return output


def triton_linear(input, weight, bias, output_shape):
    input_shape = input.shape
    weight_shape = weight.shape
    output_shape = output_shape

    # Ensure input, weight, bias, and output are on the GPU
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and output.is_cuda, "Tensors must be on CUDA."

    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output = output.contiguous()

    # Determine the number of blocks needed
    N, C_in = input_shape
    C_out, _ = weight_shape

    # Calculate the number of blocks
    num_blocks = (N * C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks2 = (C_in + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    linear_kernel[grid](input, weight, bias, output, input_shape, weight_shape, output_shape, BLOCK_SIZE=128)
    return output


def triton_relu(input, output_shape):
    input_shape = input.shape
    output_shape = output_shape

    # Ensure input and output are on the GPU
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA."

    input = input.contiguous()
    output = output.contiguous()

    # Determine the number of blocks needed
    N, C, H, W = input_shape

    # Calculate the number of blocks
    num_blocks = (N * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    relu_kernel[grid](input, output, input_shape, BLOCK_SIZE=128)
    return output


def triton_max_pool2d(input, output_shape, kernel_size, stride, padding):
    input_shape = input.shape
    output_shape = output_shape

    # Ensure input and output are on the GPU
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA."

    input = input.contiguous()
    output = output.contiguous()

    # Determine the number of blocks needed
    N, C, H, W = input_shape
    H_out = (H + 2 * padding - kernel_size + 1) // stride
    W_out = (W + 2 * padding - kernel_size + 1) // stride

    # Calculate the number of blocks
    num_blocks = (N * C * H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    max_pool2d_kernel[grid](input, output, input_shape, output_shape, kernel_size, stride, padding, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.num_classes = num_classes
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.0)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.0)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.maxpool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.maxpool5(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        return x