import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_mish_mish_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    kernel_h,
    kernel_w,
    out_h,
    out_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Get the block position in the output
    block_h = pid // out_w
    block_w = pid % out_w

    # Compute the block's position in the input
    input_h = block_h * stride_h - pad_h
    input_w = block_w * stride_w - pad_w

    # Compute the block's offset in the output
    offset_h = block_h * BLOCK_H
    offset_w = block_w * BLOCK_W

    # Compute the range of threads in the block
    thread_h = tl.arange(0, BLOCK_H)
    thread_w = tl.arange(0, BLOCK_W)

    # Compute the input and output indices
    input_indices = (input_h + thread_h[:, None])[:, None] * out_w + (input_w + thread_w[None, :]) + (thread_h[:, None] * out_w + thread_w[None, :]) * out_w
    input_indices = input_indices.flatten()
    output_indices = (offset_h + thread_h[:, None])[:, None] * out_w + (offset_w + thread_w[None, :]) + (thread_h[:, None] * out_w + thread_w[None, :]) * out_w
    output_indices = output_indices.flatten()

    # Load weights and bias
    weight = tl.load(weight_ptr + (thread_h[:, None] * kernel_w + thread_w[None, :]), mask=(thread_h[:, None] * kernel_w + thread_w[None, :]) < (kernel_h * kernel_w), other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, GROUP_SIZE), mask=tl.arange(0, GROUP_SIZE) < GROUP_SIZE, other=0.0)

    # Compute the convolution
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for k in range(GROUP_SIZE):
        input_val = tl.load(input_ptr + input_indices + k * out_w * out_w, mask=input_indices + k * out_w * out_w < input_ptr.shape[0], other=0.0)
        acc += input_val * weight[k]

    # Apply bias
    acc += bias

    # Apply Mish activation
    acc = tl.where(acc > 0, acc * tl.math.tanh(acc * 1.0 + 0.5), acc * 1.0)

    # Store the result
    tl.store(output_ptr + output_indices, acc, mask=output_indices < output_ptr.shape[0])


def triton_conv2d_mish_mish(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    kernel_h: int,
    kernel_w: int,
):
    # Ensure the input is contiguous on GPU
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Compute output dimensions
    out_h = (input.shape[2] + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (input.shape[3] + 2 * pad_w - kernel_w) // stride_w + 1

    # Allocate output tensor
    output = torch.empty((input.shape[0], weight.shape[0], out_h, out_w), dtype=input.dtype, device=input.device)

    # Determine the number of blocks
    num_blocks = (out_h * out_w + (BLOCK_H * BLOCK_W - 1)) // (BLOCK_H * BLOCK_W)

    # Launch the kernel
    grid = (num_blocks,)
    conv2d_mish_mish_kernel[grid](
        input,
        weight,
        bias,
        output,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        BLOCK_H=16,
        BLOCK_W=16,
        GROUP_SIZE=weight.shape[0],
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (kernel_size - 1) // 2

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Perform convolution
        x = triton_conv2d_mish_mish(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.kernel_size,
            self.kernel_size,
        )
        # Apply Mish activation
        x = torch.nn.functional.mish(x)
        x = torch.nn.functional.mish(x)
        return x