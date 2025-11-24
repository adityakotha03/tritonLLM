import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
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
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block index in the output
    block_idx = pid
    # Compute the output position for this block
    output_pos = block_idx * BLOCK_SIZE
    # Compute the input position corresponding to this output block
    input_pos = output_pos - padding
    # Compute the input offset for this block
    input_offset = input_pos * in_channels
    # Compute the weight offset for this block
    weight_offset = (block_idx % out_channels) * in_channels * kernel_size
    # Compute the number of elements in the block
    num_elements = min(BLOCK_SIZE, (batch_size * out_channels) - output_pos)
    # Create a range of offsets
    offsets = tl.arange(0, num_elements)
    # Load input data
    input_data = tl.load(input_ptr + input_offset + offsets, mask=offsets < (batch_size * in_channels), other=0.0)
    # Load weight data
    weight_data = tl.load(weight_ptr + weight_offset + offsets, mask=offsets < kernel_size, other=0.0)
    # Perform the convolution
    output_data = tl.dot(input_data, weight_data)
    # Store the result
    tl.store(output_ptr + output_pos + offsets, output_data, mask=offsets < num_elements)


def triton_conv_transpose1d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
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
    output = torch.empty((batch_size, out_channels, (input.size(2) - 1) * stride + 1), device=input.device, dtype=input.dtype)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose1d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Create weight and bias tensors
        weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, device=x.device, dtype=x.dtype)
        bias = torch.randn(self.out_channels, device=x.device, dtype=x.dtype) if self.bias else None

        # Perform the transposed 1D convolution using Triton kernel
        output = triton_conv_transpose1d(x, weight, bias, x.size(0), self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation)
        return output