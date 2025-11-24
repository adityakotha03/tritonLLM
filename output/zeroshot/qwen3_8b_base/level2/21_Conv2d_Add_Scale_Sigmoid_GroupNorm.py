import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor
    scale_ptr,  # Pointer to scale tensor
    stride,  # Stride of the convolution
    padding,  # Padding of the convolution
    kernel_size,  # Size of the kernel
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    height,  # Height of the input
    width,  # Width of the input
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Compute the 2D position in the output
    pid = tl.program_id(0)
    # Compute the 2D position in the input
    pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the 2D position in the output
    pos_out = pos // (width * height)
    pos_out = pos_out * (width * height) + (pos % (width * height))
    pos_out = tl.reshape(pos_out, (BLOCK_SIZE,))
    # Compute the 2D position in the input
    pos_in = pos_out + tl.arange(0, BLOCK_SIZE)
    # Compute the 2D position in the input
    pos_in = tl.reshape(pos_in, (BLOCK_SIZE,))

    # Compute the input and output indices
    out_idx = pos_out
    in_idx = pos_in

    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=in_idx < input_ptr.size, other=0.0)
    # Load weight values
    weight_val = tl.load(weight_ptr + pos_out, mask=pos_out < weight_ptr.size, other=0.0)
    # Compute the convolution
    conv_val = tl.dot(input_val, weight_val)
    # Add bias
    conv_val += tl.load(bias_ptr + pos_out // (width * height), mask=pos_out // (width * height) < bias_ptr.size, other=0.0)
    # Scale
    conv_val *= tl.load(scale_ptr + pos_out // (width * height), mask=pos_out // (width * height) < scale_ptr.size, other=1.0)
    # Apply sigmoid
    conv_val = 1.0 / (1.0 + tl.exp(-conv_val))
    # Group norm
    group_idx = pos_out // (width * height) // out_channels // GROUP_SIZE
    group_start = group_idx * GROUP_SIZE
    group_end = group_start + GROUP_SIZE
    group_mean = tl.sum(conv_val[group_start:group_end]) / GROUP_SIZE
    group_var = tl.sum((conv_val[group_start:group_end] - group_mean) ** 2) / GROUP_SIZE
    conv_val = (conv_val[group_start:group_end] - group_mean) / tl.sqrt(group_var + 1e-5)
    # Store output
    tl.store(output_ptr + out_idx, conv_val, mask=out_idx < output_ptr.size)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, stride, padding, kernel_size, out_channels, in_channels, height, width):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous()

    # Compute output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    GROUP_SIZE = 16  # Tunable parameter for group size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, bias, scale, stride, padding, kernel_size, out_channels, in_channels, height, width, BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=GROUP_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.stride = 1
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # Perform convolution
        x = triton_conv2d(x, self.weight, self.bias, self.scale, self.stride, self.padding, self.kernel_size, self.out_channels, self.in_channels, x.size(2), x.size(3))
        return x