import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input
    weight_ptr,  # Pointer to weight
    output_ptr,  # Pointer to output
    input_offset,  # Offset to start of input data
    weight_offset,  # Offset to start of weight data
    output_offset,  # Offset to start of output data
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    input_height,  # Height of input
    input_width,  # Width of input
    output_height,  # Height of output
    output_width,  # Width of output
    kernel_size,  # Size of kernel
    stride,  # Stride
    BLOCK_H: tl.constexpr,  # Block height
    BLOCK_W: tl.constexpr,  # Block width
    BLOCK_C: tl.constexpr,  # Block channels
):
    # Define block dimensions
    pid_batch = tl.program_id(0)  # Batch ID
    pid_c = tl.program_id(1)  # Channel ID
    pid_h = tl.program_id(2)  # Height ID
    pid_w = tl.program_id(3)  # Width ID

    # Compute block offsets
    start_h = pid_h * BLOCK_H
    start_w = pid_w * BLOCK_W
    start_c = pid_c * BLOCK_C

    # Load weight data
    weight_offset = weight_offset + start_c * in_channels * kernel_size * kernel_size
    weight_ptrs = weight_ptr + weight_offset
    weights = tl.load(weight_ptrs + tl.arange(0, in_channels)[:, None, None] *
                      (kernel_size * kernel_size) + tl.arange(0, kernel_size)[None, :, None] *
                      kernel_size + tl.arange(0, kernel_size)[None, None, :],
                      mask=(tl.arange(0, in_channels)[:, None, None] < in_channels) &
                           (tl.arange(0, kernel_size)[None, :, None] < kernel_size) &
                           (tl.arange(0, kernel_size)[None, None, :] < kernel_size),
                      other=0.0)

    # Compute output indices
    input_h = start_h // stride
    input_w = start_w // stride
    output_h = start_h + tl.arange(0, BLOCK_H)
    output_w = start_w + tl.arange(0, BLOCK_W)
    input_h = input_h + tl.arange(0, BLOCK_H) // stride
    input_w = input_w + tl.arange(0, BLOCK_W) // stride

    # Load input data
    input_offset = input_offset + pid_batch * in_channels * input_height * input_width
    input_ptrs = input_ptr + input_offset + (input_h * input_width + input_w) * in_channels
    inputs = tl.load(input_ptrs + tl.arange(0, in_channels)[:, None, None],
                     mask=(tl.arange(0, in_channels)[:, None, None] < in_channels),
                     other=0.0)

    # Perform convolution transpose
    output = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    for c in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                input_h_offset = input_h + kh
                input_w_offset = input_w + kw
                input_h_mask = (input_h_offset < input_height)
                input_w_mask = (input_w_offset < input_width)
                input_ptr_offset = input_ptrs + c + (input_h_offset * input_width + input_w_offset) * in_channels
                input_vals = tl.load(input_ptr_offset, mask=input_h_mask[:, None, None] &
                                                   input_w_mask[None, :, None], other=0.0)
                weight_val = weights[c, kh, kw]
                output += input_vals[:, :, None] * weight_val[None, None, :]

    # Store output
    output_offset = output_offset + pid_batch * out_channels * output_height * output_width
    output_ptrs = output_ptr + output_offset + (output_h * output_width + output_w) * out_channels
    output_ptrs = output_ptrs + tl.arange(0, BLOCK_C)[None, None, :]
    tl.store(output_ptrs, output, mask=(tl.arange(0, BLOCK_H)[:, None, None] < output_height) &
                                       (tl.arange(0, BLOCK_W)[None, :, None] < output_width) &
                                       (tl.arange(0, BLOCK_C)[None, None, :] < out_channels))


@triton.jit
def add_and_min_kernel(
    x_ptr,  # Pointer to input
    add_value,  # Value to add
    output_ptr,  # Pointer to output
    n_elements,  # Number of elements
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
    # Add value
    x += add_value
    # Apply min with zero
    x = tl.minimum(x, 0.0)
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input
    output_ptr,  # Pointer to output
    n_elements,  # Number of elements
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
    # Apply GELU activation
    x = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def multiply_kernel(
    x_ptr,  # Pointer to input
    multiply_value,  # Value to multiply
    output_ptr,  # Pointer to output
    n_elements,  # Number of elements
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
    # Multiply by value
    x *= multiply_value
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_conv_transpose(x, weight, output_height, output_width, kernel_size, stride):
    """
    Triton-based transposed convolution wrapper.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    out = torch.empty(x.size(0), weight.size(0), output_height, output_width, device=x.device, dtype=x.dtype)

    # Grid configuration
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, _, _ = weight.shape
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 32

    # Calculate grid size
    grid = lambda meta: (batch_size, (out_channels + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
                         (output_height + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
                         (output_width + meta["BLOCK_W"] - 1) // meta["BLOCK_W"])

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, out,
        0, 0, 0,
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )

    return out


def triton_add_and_min(x, add_value):
    """
    Triton-based addition and min(0) wrapper.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Grid configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    add_and_min_kernel[grid](x, add_value, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


def triton_gelu(x):
    """
    Triton-based GELU wrapper.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Grid configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


def triton_multiply(x, multiply_value):
    """
    Triton-based multiplication wrapper.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Grid configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    multiply_kernel[grid](x, multiply_value, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Apply transposed convolution with Triton kernel
        output_height = (x.shape[2] - 1) * stride + kernel_size
        output_width = (x.shape[3] - 1) * stride + kernel_size
        x = triton_conv_transpose(x, self.conv_transpose.weight, output_height, output_width, kernel_size, stride)
        # Apply add and min(0)
        x = triton_add_and_min(x, self.add_value)
        # Apply GELU
        x = triton_gelu(x)
        # Apply multiplication
        x = triton_multiply(x, self.multiply_value)
        return x