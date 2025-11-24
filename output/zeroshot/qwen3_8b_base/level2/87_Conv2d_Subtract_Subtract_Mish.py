import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # Extract the block index
    pid = tl.program_id(0)
    # Calculate the output position
    out_h = pid // (input_shape[3] // stride)
    out_w = pid % (input_shape[3] // stride)
    # Calculate the input position
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding
    # Initialize output value
    out_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # Iterate over the kernel
    for k in range(kernel_size):
        for c in range(input_shape[1]):
            # Calculate the input offset
            in_offset = (in_h + k) * input_shape[3] + in_w
            # Load input values
            input_val = tl.load(input_ptr + in_offset + c * input_shape[3], mask=in_offset + c * input_shape[3] < input_shape[2] * input_shape[3], other=0.0)
            # Multiply by weight and accumulate
            weight_val = tl.load(weight_ptr + c * input_shape[1] + k * input_shape[1], other=0.0)
            out_val += input_val * weight_val
    # Apply Mish activation
    out_val = out_val * tl.math.tanh(tl.math.softplus(out_val))
    # Store output
    output_ptr[pid] = out_val


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    Triton kernel for Conv2D with Mish activation.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty((input.shape[0], weight.shape[0], input.shape[2] - 2 * padding, input.shape[3] - 2 * padding), dtype=input.dtype, device=input.device)

    # Calculate number of output elements
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input.shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def subtract_and_mish_kernel(
    input_ptr,  # Pointer to input tensor
    subtract1_ptr,  # Pointer to subtract value 1
    subtract2_ptr,  # Pointer to subtract value 2
    output_ptr,  # Pointer to output tensor
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
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Load subtract values
    s1 = tl.load(subtract1_ptr, other=0.0)
    s2 = tl.load(subtract2_ptr, other=0.0)
    # Subtract values
    x = x - s1 - s2
    # Apply Mish activation
    x = x * tl.math.tanh(tl.math.softplus(x))
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_subtract_and_mish(x: torch.Tensor, s1: float, s2: float):
    """
    Triton kernel for subtracting two values and applying Mish activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    output = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    subtract_and_mish_kernel[grid](x, torch.tensor(s1).cuda(), torch.tensor(s2).cuda(), output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        # Replace the Conv2D with Triton kernel
        x = triton_conv2d(x, self.weight, self.stride, self.padding)
        # Replace the subtraction and Mish with Triton kernel
        x = triton_subtract_and_mish(x, self.subtract_value_1, self.subtract_value_2)
        return x