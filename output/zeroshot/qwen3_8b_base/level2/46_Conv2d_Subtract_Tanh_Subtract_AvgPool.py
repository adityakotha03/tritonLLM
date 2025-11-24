import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    stride,  # Stride of the convolution
    kernel_size,  # Kernel size
    BLOCK_SIZE: tl.constexpr,
):
    # Get the position in the output tensor (output_h, output_w, output_c)
    pid = tl.program_id(0)
    # Get the position in the input tensor (input_h, input_w, input_c)
    input_h = pid // (BLOCK_SIZE * BLOCK_SIZE)
    input_w = (pid // BLOCK_SIZE) % BLOCK_SIZE
    input_c = pid % BLOCK_SIZE

    # Compute the offset in the input tensor
    input_offset = input_c * (height * width) + input_h * width + input_w
    input_ptr_base = input_ptr + input_offset

    # Compute the output position (output_h, output_w)
    output_h = input_h
    output_w = input_w

    # Compute the weight offset
    weight_offset = input_c * (kernel_size * kernel_size) + (input_h % kernel_size) * kernel_size + (input_w % kernel_size)
    weight_ptr_base = weight_ptr + weight_offset

    # Compute the output offset
    output_offset = output_h * width + output_w
    output_ptr_base = output_ptr + output_offset

    # Load the input value
    input_val = tl.load(input_ptr_base)

    # Load the weight value
    weight_val = tl.load(weight_ptr_base)

    # Compute the output value
    output_val = input_val * weight_val

    # Store the output value
    tl.store(output_ptr_base, output_val)


@triton.jit
def subtract_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    value,  # Value to subtract
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
    # Perform the subtraction
    out = x - value
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def tanh_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
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
    # Compute tanh
    out = tl.tanh(x)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def avg_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    kernel_size,  # Kernel size
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Get the position in the input tensor (input_h, input_w, input_c)
    pid = tl.program_id(0)
    input_h = pid // (BLOCK_SIZE * BLOCK_SIZE)
    input_w = (pid // BLOCK_SIZE) % BLOCK_SIZE
    input_c = pid % BLOCK_SIZE

    # Compute the input offset
    input_offset = input_c * (height * width) + input_h * width + input_w
    input_ptr_base = input_ptr + input_offset

    # Compute the output position (output_h, output_w)
    output_h = input_h // kernel_size
    output_w = input_w // kernel_size

    # Compute the output offset
    output_offset = output_h * width + output_w
    output_ptr_base = output_ptr + output_offset

    # Load the input value
    input_val = tl.load(input_ptr_base)

    # Accumulate the sum
    sum_val = input_val
    count = 1

    # Iterate over the kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            offset = input_c * (height * width) + (input_h + i) * width + (input_w + j)
            val = tl.load(input_ptr + offset)
            sum_val += val
            count += 1

    # Compute average
    avg_val = sum_val / count

    # Store the result
    tl.store(output_ptr_base, avg_val)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, kernel_size: int):
    """
    Custom Triton kernel for 2D convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, stride, kernel_size, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_subtract(x: torch.Tensor, value: float):
    """
    Custom Triton kernel for subtraction.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    subtract_kernel[grid](x, out, value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_tanh(x: torch.Tensor):
    """
    Custom Triton kernel for tanh activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_avg_pool2d(input: torch.Tensor, kernel_size: int):
    """
    Custom Triton kernel for average pooling.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty((input.size(0), input.size(1), input.size(2) // kernel_size, input.size(3) // kernel_size), device=input.device)

    # Number of elements in the tensor
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    avg_pool2d_kernel[grid](input, output, kernel_size, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        # Custom convolution
        x = triton_conv2d(x, self.weight, 1, self.kernel_size)
        # Custom subtraction
        x = triton_subtract(x, self.subtract1_value)
        # Custom tanh
        x = triton_tanh(x)
        # Custom subtraction
        x = triton_subtract(x, self.subtract2_value)
        # Custom average pooling
        x = triton_avg_pool2d(x, self.kernel_size_pool)
        return x