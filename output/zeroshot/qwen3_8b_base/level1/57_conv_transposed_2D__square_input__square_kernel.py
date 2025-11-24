import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Compute the batch index
    batch_idx = pid // (out_channels * (kernel_size * kernel_size))
    # Compute the output channel index
    out_ch_idx = (pid % (out_channels * (kernel_size * kernel_size))) // (kernel_size * kernel_size)
    # Compute the spatial position within the kernel
    kernel_pos = (pid % (kernel_size * kernel_size))
    # Compute the output spatial position
    pos = tl.arange(0, BLOCK_SIZE)
    # Compute the input spatial position
    input_pos = pos + padding
    # Compute the output spatial position
    output_pos = input_pos + (stride - 1) * pos
    # Compute the input channel index
    in_ch_idx = tl.arange(0, in_channels)
    # Compute the input offset
    input_offset = batch_idx * in_channels * height * width + in_ch_idx * height * width + input_pos * width
    # Compute the weight offset
    weight_offset = out_ch_idx * in_channels * kernel_size * kernel_size + kernel_pos
    # Compute the output offset
    output_offset = batch_idx * out_channels * height_out * width_out + out_ch_idx * height_out * width_out + output_pos * width_out
    # Load input and weight
    input_val = tl.load(input_ptr + input_offset, mask=input_offset < input_ptr.size, other=0.0)
    weight_val = tl.load(weight_ptr + weight_offset, mask=weight_offset < weight_ptr.size, other=0.0)
    # Perform the convolution
    output_val = input_val * weight_val
    # Store the result
    tl.store(output_ptr + output_offset, output_val, mask=output_offset < output_ptr.size)


def triton_conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, output_padding: int):
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

    # Compute output dimensions
    height_out = (input.size(2) - 1) * stride + kernel_size - 2 * padding + output_padding
    width_out = (input.size(3) - 1) * stride + kernel_size - 2 * padding + output_padding

    # Prepare output tensor
    output = torch.empty((input.size(0), out_channels, height_out, width_out), device=input.device, dtype=input.dtype)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](input, weight, output, input.size(0), input.size(1), output.size(1), kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Create weight and bias tensors
        weight = torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size, device=x.device, dtype=x.dtype)
        bias = torch.randn(self.out_channels, device=x.device, dtype=x.dtype) if self.bias else None

        # Perform the transposed 2D convolution using the custom Triton kernel
        output = triton_conv_transpose2d(x, weight, bias, self.stride, self.padding, self.output_padding)
        return output