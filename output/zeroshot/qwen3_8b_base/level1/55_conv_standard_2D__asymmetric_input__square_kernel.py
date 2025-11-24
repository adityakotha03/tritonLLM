import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the thread index
    pid = tl.program_id(0)
    # Compute the block index in the output
    out_h = pid // (out_channels // BLOCK_SIZE)
    out_w = pid % (out_channels // BLOCK_SIZE)
    # Compute the output position
    out_y = out_h * stride
    out_x = out_w * stride
    # Compute the input position
    in_y = out_y - padding
    in_x = out_x - padding
    # Compute the input channel index
    in_c = tl.program_id(1)
    # Compute the output channel index
    out_c = tl.program_id(2)
    # Compute the input offset
    in_offset = tl.arange(0, BLOCK_SIZE)
    # Compute the weight offset
    weight_offset = tl.arange(0, BLOCK_SIZE)
    # Compute the input and weight indices
    input_indices = (in_offset + in_y * width + in_x * height) * in_channels + in_c
    weight_indices = (weight_offset + out_c * in_channels) * kernel_size * kernel_size + out_c
    # Load input and weight values
    input_values = tl.load(input_ptr + input_indices, mask=input_indices < input_ptr.size, other=0.0)
    weight_values = tl.load(weight_ptr + weight_indices, mask=weight_indices < weight_ptr.size, other=0.0)
    # Perform the convolution
    output_values = tl.dot(input_values, weight_values)
    # Store the result
    tl.store(output_ptr + out_c * out_channels + out_y * width + out_x, output_values)


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
    output = torch.empty_like(input)

    # Compute output dimensions
    batch_size = input.size(0)
    in_channels = input.size(1)
    height = input.size(2)
    width = input.size(3)
    out_channels = weight.size(0)
    kernel_size = weight.size(1)
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Number of elements in the tensor
    n_elements = out_channels * out_h * out_w

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], in_channels, out_channels)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=128)
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
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Perform the convolution
        output = triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return output