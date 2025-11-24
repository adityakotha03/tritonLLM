import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_channels,  # Input channels
    out_channels,  # Output channels
    depth,  # Depth of input
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    output_padding,  # Output padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index along the batch dimension
    batch_idx = tl.program_id(0)
    # Compute the block index along the output depth dimension
    out_depth_idx = tl.program_id(1)
    # Compute the block index along the output height dimension
    out_height_idx = tl.program_id(2)
    # Compute the block index along the output width dimension
    out_width_idx = tl.program_id(3)

    # Compute the offset in the output tensor
    out_offset = (
        batch_idx * out_channels * depth * height * width +
        out_depth_idx * out_channels * height * width +
        out_height_idx * out_channels * width +
        out_width_idx * out_channels
    )

    # Compute the offset in the input tensor
    in_offset = (
        batch_idx * in_channels * depth * height * width +
        tl.arange(0, BLOCK_SIZE) * height * width +
        tl.arange(0, BLOCK_SIZE) * width +
        tl.arange(0, BLOCK_SIZE)
    )

    # Compute the offset in the weight tensor
    weight_offset = (
        tl.arange(0, out_channels) * in_channels * kernel_size * kernel_size * kernel_size +
        tl.arange(0, BLOCK_SIZE) * kernel_size * kernel_size * kernel_size +
        tl.arange(0, BLOCK_SIZE) * kernel_size * kernel_size +
        tl.arange(0, BLOCK_SIZE) * kernel_size +
        tl.arange(0, BLOCK_SIZE)
    )

    # Compute the offset in the bias tensor
    bias_offset = tl.arange(0, out_channels)

    # Compute the output index for each thread
    output_idx = out_offset + tl.arange(0, BLOCK_SIZE)

    # Load input values
    input_vals = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < (depth * height * width), other=0.0)
    # Load weight values
    weight_vals = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < (in_channels * kernel_size * kernel_size * kernel_size), other=0.0)
    # Load bias values
    bias_vals = tl.load(bias_ptr + bias_offset, mask=tl.arange(0, out_channels) < out_channels, other=0.0)

    # Compute the convolution
    output_vals = tl.dot(input_vals, weight_vals) + bias_vals

    # Store the output values
    tl.store(output_ptr + output_idx, output_vals, mask=tl.arange(0, BLOCK_SIZE) < (depth * height * width))


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def multiply_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise multiplication
    out = x * y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose3d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding, output_padding):
    """
    This function wraps the Triton kernel call for 3D transposed convolution.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Calculate output dimensions
    out_depth = (depth - 1) * stride + kernel_size - 2 * padding + output_padding
    out_height = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_width = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the grid (blocks) needed
    grid = lambda meta: (batch_size, out_depth, out_height, out_width)

    # Launch the Triton kernel
    conv_transpose3d_kernel[grid](input, weight, bias, output, batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_add(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call for elementwise addition.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_multiply(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call for elementwise multiplication.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    multiply_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape

    def forward(self, x):
        # Perform 3D transposed convolution
        weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size, device=x.device)
        bias = torch.randn(self.bias_shape, device=x.device)
        x = triton_conv_transpose3d(x, weight, bias, x.size(0), self.in_channels, self.out_channels, x.size(2), x.size(3), x.size(4), self.kernel_size, self.stride, self.padding, self.output_padding)

        # Clone original x for residual additions
        original_x = x.clone().detach()

        # First residual add
        x = triton_add(x, self.bias)

        # Second residual add
        x = triton_add(x, original_x)

        # Elementwise multiplication
        x = triton_multiply(x, original_x)

        # Third residual add
        x = triton_add(x, original_x)

        return x