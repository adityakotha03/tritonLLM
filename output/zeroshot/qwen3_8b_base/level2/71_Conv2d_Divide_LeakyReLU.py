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
    weight_shape,  # (out_channels, in_channels, kernel_size, kernel_size)
    output_shape,  # (batch, out_channels, height, width)
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    # Get the thread index within the block
    thread_idx = tl.program_id(1)
    # Compute the block's position in the output
    out_batch = block_idx // (output_shape[1] * output_shape[2] * output_shape[3])
    out_channel = (block_idx // (output_shape[2] * output_shape[3])) % output_shape[1]
    out_h = (block_idx // output_shape[3]) % output_shape[2]
    out_w = block_idx % output_shape[3]

    # Compute the input coordinates
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Compute the offset in the input tensor
    input_offset = (out_batch * input_shape[1] + out_channel) * input_shape[2] * input_shape[3] + in_h * input_shape[3] + in_w

    # Compute the weight offset
    weight_offset = out_channel * weight_shape[1] * weight_shape[2] * weight_shape[3] + thread_idx

    # Initialize the output value
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the kernel
    for k in range(weight_shape[2]):
        for l in range(weight_shape[3]):
            # Compute the input offset for this kernel position
            in_h_k = in_h + k
            in_w_l = in_w + l
            input_offset_kl = input_offset + in_h_k * input_shape[3] + in_w_l

            # Load input value
            input_val = tl.load(input_ptr + input_offset_kl, mask=input_offset_kl < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=0.0)

            # Load weight value
            weight_val = tl.load(weight_ptr + weight_offset, other=0.0)

            # Multiply and accumulate
            acc += input_val * weight_val

    # Store the result
    output_offset = (out_batch * output_shape[1] + out_channel) * output_shape[2] * output_shape[3] + out_h * output_shape[3] + out_w
    tl.store(output_ptr + output_offset, acc)


@triton.jit
def leaky_relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
    negative_slope: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Apply Leaky ReLU
    y = tl.where(x >= 0, x, x * negative_slope)
    # Store the result
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def divide_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
    divisor: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Perform the division
    y = x / divisor
    # Store the result
    tl.store(output_ptr + offsets, y, mask=mask)


def triton_conv2d(input, weight, stride, padding):
    # Ensure the input and weight are on the GPU
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Compute the output shape
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1
    output_shape = (batch, out_channels, out_h, out_w)

    # Prepare output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Launch the Triton kernel
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    conv2d_kernel[grid](input, weight, output, input.shape, weight.shape, output.shape, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_leaky_relu(input, negative_slope):
    # Ensure the input is on the GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Determine the number of blocks needed
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Launch the Triton kernel
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    leaky_relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, negative_slope=negative_slope)
    return output


def triton_divide(input, divisor):
    # Ensure the input is on the GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Determine the number of blocks needed
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Launch the Triton kernel
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    divide_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, divisor=divisor)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = divisor

    def forward(self, x):
        # Perform convolution with Triton kernel
        x = triton_conv2d(x, self.weight, stride=1, padding=(self.kernel_size - 1) // 2)
        # Perform division with Triton kernel
        x = triton_divide(x, self.divisor)
        # Perform Leaky ReLU with Triton kernel
        x = triton_leaky_relu(x, negative_slope=0.01)
        return x