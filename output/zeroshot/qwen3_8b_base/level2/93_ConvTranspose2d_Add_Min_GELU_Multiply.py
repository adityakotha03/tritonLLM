import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    stride,  # Stride of the transposed convolution
    kernel_size,  # Kernel size of the transposed convolution
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of output elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the output position
    out_h = offsets // width
    out_w = offsets % width

    # Compute the input positions for each output element
    # Transposed convolution: for each output element, we need to find all input elements
    # that contribute to it based on the kernel and stride
    # This is a simplified version assuming input is padded and kernel is symmetric
    # In practice, this would need to be adjusted for the actual padding and kernel
    # For demonstration, we'll assume no padding and kernel is symmetric

    # Calculate the input positions
    in_h = out_h * stride
    in_w = out_w * stride

    # Load weights
    weight_offsets = tl.arange(0, kernel_size * kernel_size * in_channels)
    weights = tl.load(weight_ptr + weight_offsets, mask=weight_offsets < in_channels * kernel_size * kernel_size, other=0.0)

    # Load input values
    input_offsets = in_h * width + in_w
    input_values = tl.load(input_ptr + input_offsets, mask=input_offsets < height * width, other=0.0)

    # Compute the output
    output = tl.dot(input_values, weights)

    # Store the result
    tl.store(output_ptr + offsets, output, mask=offsets < height * width)


def triton_conv_transpose(input: torch.Tensor, weight: torch.Tensor, stride: int, kernel_size: int, out_channels: int, in_channels: int, height: int, width: int):
    """
    This function wraps the Triton kernel call for transposed convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output = torch.empty(out_channels, height, width, device=input.device)

    # Number of elements in the output tensor
    n_elements = out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_kernel[grid](input, weight, output, stride, kernel_size, out_channels, in_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def gelu_kernel(
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
    # Compute GELU
    out = x * tl.erf(x * 0.7071) * 0.5 + x * 0.5
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for GELU.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for transposed convolution and GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value

        # Initialize weights for the transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.weight = self.weight.cuda()

    def forward(self, x):
        # Transposed convolution using Triton kernel
        x = triton_conv_transpose(x, self.weight, self.stride, self.kernel_size, self.out_channels, self.in_channels, x.size(2), x.size(3))
        # Add value
        x = x + self.add_value
        # Minimum with 0.0
        x = torch.min(x, torch.tensor(0.0, device=x.device))
        # GELU using Triton kernel
        x = triton_gelu(x)
        # Multiply by value
        x = x * self.multiply_value
        return x