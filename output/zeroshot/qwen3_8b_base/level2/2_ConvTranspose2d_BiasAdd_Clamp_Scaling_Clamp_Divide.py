import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    stride,  # Stride of the transposed convolution
    padding,  # Padding of the transposed convolution
    output_padding,  # Output padding of the transposed convolution
    kernel_size,  # Kernel size of the transposed convolution
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    height,  # Height of input tensor
    width,  # Width of input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_channels * height * width

    # Compute the position in the output tensor
    out_idx = offsets
    out_h = out_idx // (out_channels * width)
    out_w = (out_idx % (out_channels * width)) // out_channels
    out_c = out_idx % out_channels

    # Compute the corresponding input position
    # For transposed convolution, we need to reverse the operation
    # Input size = (output size - 1) * stride - 2 * padding + kernel_size
    # So input_h = (out_h - 1) * stride - 2 * padding + kernel_size
    # But for simplicity, we assume the input is correctly sized

    # Calculate input coordinates
    input_h = (out_h - 1) * stride - 2 * padding + kernel_size
    input_w = (out_w - 1) * stride - 2 * padding + kernel_size

    # Compute the input index
    input_idx = out_c * in_channels * input_h * input_w + tl.arange(0, BLOCK_SIZE)
    input_mask = input_idx < out_channels * in_channels * input_h * input_w

    # Load input values
    x = tl.load(x_ptr + input_idx, mask=input_mask, other=0.0)
    weight = tl.load(weight_ptr + out_c * in_channels * kernel_size * kernel_size + tl.arange(0, BLOCK_SIZE), mask=input_mask, other=0.0)
    bias = tl.load(bias_ptr + out_c, other=0.0)

    # Perform the convolution
    out = tl.dot(x, weight) + bias
    tl.store(out_ptr + out_idx, out, mask=mask)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride, padding, output_padding, kernel_size, out_channels, in_channels, height, width):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out = torch.empty((out_channels, height, width), device=x.device, dtype=x.dtype)

    # Number of elements in the output tensor
    n_elements = out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_kernel[grid](x, weight, bias, out, stride, padding, output_padding, kernel_size, out_channels, in_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernels for transposed convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        self.scaling_factor = scaling_factor

        # Initialize bias
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Perform transposed convolution with custom Triton kernel
        x = triton_conv_transpose(
            x, 
            torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).cuda(), 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.kernel_size, 
            self.out_channels, 
            self.in_channels, 
            x.size(2), 
            x.size(3)
        )
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x * self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
        return x