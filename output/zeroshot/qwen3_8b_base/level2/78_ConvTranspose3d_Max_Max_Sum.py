import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose_kernel(
    input_ptr, output_ptr,
    input_shape, output_shape,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_shape

    # Compute the output index
    out_idx = offsets
    # Compute the input indices
    in_idx = tl.where(
        (out_idx // (output_shape // input_shape)) < input_shape,
        (out_idx // (output_shape // input_shape)) * (stride ** 3) +
        (out_idx % (output_shape // input_shape) // (output_shape // input_shape // (stride ** 3))) * (stride ** 2) +
        (out_idx % (output_shape // input_shape) % (output_shape // input_shape // (stride ** 3))) // (output_shape // input_shape // (stride ** 3) // (stride ** 2)) * stride +
        (out_idx % (output_shape // input_shape) % (output_shape // input_shape // (stride ** 3) // (stride ** 2))),
        0
    )

    # Load input
    input_val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)

    # Compute output
    output_val = input_val

    # Store output
    tl.store(output_ptr + out_idx, output_val, mask=mask)

def triton_conv_transpose(input, output, input_shape, output_shape, kernel_size, stride, padding):
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 128
    num_blocks = (output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    conv_transpose_kernel[grid](input, output, input_shape, output_shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output

@triton.jit
def max_pool_kernel(
    input_ptr, output_ptr,
    input_shape, output_shape,
    kernel_size, stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_shape

    # Compute output index
    out_idx = offsets
    # Compute input indices
    in_idx = tl.where(
        (out_idx // (output_shape // input_shape)) < input_shape,
        (out_idx // (output_shape // input_shape)) * (stride ** 3) +
        (out_idx % (output_shape // input_shape) // (output_shape // input_shape // (stride ** 3))) * (stride ** 2) +
        (out_idx % (output_shape // input_shape) % (output_shape // input_shape // (stride ** 3) // (stride ** 2))) // (output_shape // input_shape // (stride ** 3) // (stride ** 2) // (stride ** 2)) * stride +
        (out_idx % (output_shape // input_shape) % (output_shape // input_shape // (stride ** 3) // (stride ** 2) // (stride ** 2))),
        0
    )

    # Load input
    input_val = tl.load(input_ptr + in_idx, mask=mask, other=-float('inf'))

    # Compute max
    output_val = tl.max(input_val, axis=0)

    # Store output
    tl.store(output_ptr + out_idx, output_val, mask=mask)

def triton_max_pool(input, output, input_shape, output_shape, kernel_size, stride):
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 128
    num_blocks = (output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    max_pool_kernel[grid](input, output, input_shape, output_shape, kernel_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return output

@triton.jit
def sum_kernel(
    input_ptr, output_ptr,
    input_shape, output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_shape

    # Compute input index
    in_idx = tl.where(
        (offsets // (output_shape // input_shape)) < input_shape,
        (offsets // (output_shape // input_shape)) * (output_shape // input_shape // input_shape) +
        (offsets % (output_shape // input_shape) // (output_shape // input_shape // input_shape)) * (output_shape // input_shape // input_shape // input_shape) +
        (offsets % (output_shape // input_shape) % (output_shape // input_shape // input_shape // input_shape)),
        0
    )

    # Load input
    input_val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)

    # Compute sum
    output_val = tl.sum(input_val, axis=0)

    # Store output
    tl.store(output_ptr + offsets, output_val, mask=mask)

def triton_sum(input, output, input_shape, output_shape):
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 128
    num_blocks = (output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    sum_kernel[grid](input, output, input_shape, output_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Custom Triton ConvTranspose3d
        input_shape = (self.in_channels, x.size(2), x.size(3), x.size(4))
        output_shape = (self.out_channels, x.size(2) * self.stride ** 3, x.size(3) * self.stride ** 3, x.size(4) * self.stride ** 3)
        output = torch.zeros(output_shape, device=x.device)
        output = triton_conv_transpose(x, output, input_shape, output_shape, self.kernel_size, self.stride, self.padding)

        # Custom Triton MaxPool3d (kernel_size=2)
        input_shape = output.size()
        output_shape = (output.size(0), output.size(1) // 2, output.size(2) // 2, output.size(3) // 2)
        output = torch.zeros(output_shape, device=x.device)
        output = triton_max_pool(output, output, input_shape, output_shape, 2, 2)

        # Custom Triton MaxPool3d (kernel_size=3)
        input_shape = output.size()
        output_shape = (output.size(0), output.size(1) // 3, output.size(2) // 3, output.size(3) // 3)
        output = torch.zeros(output_shape, device=x.device)
        output = triton_max_pool(output, output, input_shape, output_shape, 3, 3)

        # Custom Triton Sum
        input_shape = output.size()
        output_shape = (1, output.size(1), output.size(2), output.size(3))
        output = torch.zeros(output_shape, device=x.device)
        output = triton_sum(output, output, input_shape, output_shape)

        return output