import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, height, width)
    weight_shape,  # (out_channels, in_channels, kernel, kernel)
    output_shape,  # (batch, out_channels, height, width)
    stride,  # Stride of the convolution
    pad,  # Padding added to both sides of the input
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    # Get the thread index within the block
    thread_idx = tl.program_id(1)
    # Calculate the output position
    out_h = block_idx // output_shape[2]
    out_w = block_idx % output_shape[2]
    # Calculate the input position
    in_h = out_h * stride - pad
    in_w = out_w * stride - pad
    # Calculate the output channel
    out_c = thread_idx
    # Calculate the input channel
    in_c = tl.arange(0, weight_shape[1])
    # Calculate the kernel positions
    kernel_h = tl.arange(0, weight_shape[2])
    kernel_w = tl.arange(0, weight_shape[3])
    # Compute the input indices
    in_idx = (
        tl.arange(0, output_shape[0]) * input_shape[1] * input_shape[2] * input_shape[3]
        + tl.arange(0, weight_shape[1]) * input_shape[2] * input_shape[3]
        + in_h * input_shape[3]
        + in_w
    )
    # Compute the weight indices
    weight_idx = (
        out_c * weight_shape[1] * weight_shape[2] * weight_shape[3]
        + in_c * weight_shape[2] * weight_shape[3]
        + kernel_h * weight_shape[3]
        + kernel_w
    )
    # Compute the output indices
    out_idx = (
        tl.arange(0, output_shape[0]) * output_shape[1] * output_shape[2] * output_shape[3]
        + out_c * output_shape[2] * output_shape[3]
        + out_h * output_shape[3]
        + out_w
    )
    # Load input and weight
    input_vals = tl.load(input_ptr + in_idx, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=0.0)
    weight_vals = tl.load(weight_ptr + weight_idx, mask=weight_idx < weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3], other=0.0)
    # Compute the convolution
    output_vals = tl.sum(input_vals * weight_vals, axis=0)
    # Store the result
    tl.store(output_ptr + out_idx, output_vals, mask=out_idx < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3])


@triton.jit
def min_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, height, width)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    # Get the thread index within the block
    thread_idx = tl.program_id(1)
    # Calculate the output position
    out_h = block_idx // input_shape[2]
    out_w = block_idx % input_shape[2]
    # Calculate the input position
    in_h = out_h
    in_w = out_w
    # Calculate the input channel
    in_c = tl.arange(0, input_shape[1])
    # Compute the input indices
    in_idx = (
        tl.arange(0, input_shape[0]) * input_shape[1] * input_shape[2] * input_shape[3]
        + in_c * input_shape[2] * input_shape[3]
        + in_h * input_shape[3]
        + in_w
    )
    # Compute the output indices
    out_idx = (
        tl.arange(0, input_shape[0]) * input_shape[1] * input_shape[2] * input_shape[3]
        + out_h * input_shape[3]
        + out_w
    )
    # Load input values
    input_vals = tl.load(input_ptr + in_idx, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=0.0)
    # Compute the minimum
    min_val = tl.min(input_vals, axis=0)
    # Store the result
    tl.store(output_ptr + out_idx, min_val, mask=out_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3])


def triton_conv2d(input, weight, stride, pad):
    # Ensure input and weight are on GPU
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    # Create output tensor
    output = torch.empty_like(input)
    # Calculate the output shape
    out_h = (input.shape[2] + 2 * pad - weight.shape[2]) // stride + 1
    out_w = (input.shape[3] + 2 * pad - weight.shape[3]) // stride + 1
    output_shape = (input.shape[0], weight.shape[0], out_h, out_w)
    # Determine the number of blocks needed
    grid = lambda meta: ((output_shape[0] * output_shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (weight_shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input.shape, weight.shape, output_shape, stride, pad, BLOCK_SIZE=128)
    return output


def triton_min(input):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    # Ensure input is contiguous
    input = input.contiguous()
    # Create output tensor
    output = torch.empty((input.shape[0], input.shape[2], input.shape[3]), device=input.device)
    # Calculate the input shape
    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    # Determine the number of blocks needed
    grid = lambda meta: ((input_shape[0] * input_shape[2] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (input_shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    min_kernel[grid](input, output, input_shape, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        # Custom Triton-based convolution
        x = triton_conv2d(x, torch.randn((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), device=x.device), stride=1, pad=(self.kernel_size - 1) // 2)
        # Scale the output
        x = x * self.scale_factor
        # Custom Triton-based minimum operation
        x = triton_min(x)
        return x