import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch_size, in_channels, height, width)
    kernel_size,  # Size of the square kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch, channel, height, and width indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    width_idx = tl.program_id(3)

    # Calculate the output dimensions
    batch_size, in_channels, height, width = input_shape
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Compute the output channel index
    out_channel_idx = channel_idx % out_channels

    # Compute the input channel index
    in_channel_idx = channel_idx // out_channels * groups + (channel_idx % out_channels) // groups

    # Compute the starting position in input
    input_offset = batch_idx * in_channels * height * width + in_channel_idx * height * width
    input_offset += height_idx * width + width_idx

    # Compute the starting position in weight
    weight_offset = out_channel_idx * in_channels * kernel_size * kernel_size + in_channel_idx * kernel_size * kernel_size
    weight_offset += (kernel_size // 2) * kernel_size + (kernel_size // 2)

    # Compute the output offset
    output_offset = batch_idx * out_channels * out_height * out_width + out_channel_idx * out_height * out_width
    output_offset += height_idx * out_width + width_idx

    # Initialize the output value
    out_val = 0.0

    # Iterate over the kernel
    for k in range(kernel_size):
        for j in range(kernel_size):
            # Compute the input position with dilation and padding
            in_h = height_idx + (k - kernel_size // 2) * dilation
            in_w = width_idx + (j - kernel_size // 2) * dilation

            # Apply padding
            if in_h < 0 or in_h >= height or in_w < 0 or in_w >= width:
                continue

            # Compute the input offset
            input_offset_k = input_offset + in_h * width + in_w

            # Load the input value
            input_val = tl.load(input_ptr + input_offset_k, mask=(in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width), other=0.0)

            # Load the weight value
            weight_val = tl.load(weight_ptr + weight_offset + k * kernel_size + j, other=0.0)

            # Multiply and accumulate
            out_val += input_val * weight_val

    # Store the output value
    tl.store(output_ptr + output_offset, out_val)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (batch_size, out_channels, out_height, out_width)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, (batch_size, in_channels, height, width), kernel_size, stride, padding, dilation, out_channels, in_channels, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create weight and bias tensors
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        weight = weight.cuda()
        bias = torch.nn.Parameter(torch.randn(self.out_channels)) if self.bias else None
        bias = bias.cuda() if bias is not None else None

        # Perform the convolution
        output = triton_conv2d(x, weight, bias, self.stride, self.padding, self.dilation)
        return output