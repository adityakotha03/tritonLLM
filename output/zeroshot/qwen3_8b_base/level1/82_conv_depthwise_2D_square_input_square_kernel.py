import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_start = pid * BLOCK_SIZE
    # Compute the block index in the output
    block_idx = block_start // (width * height)
    # Compute the block position in the output
    block_pos = block_start % (width * height)
    # Compute the output coordinates
    out_h = block_pos // width
    out_w = block_pos % width
    # Compute the input coordinates
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding
    # Compute the input offset
    in_offset = (block_idx * in_channels * height * width) + (in_h * width + in_w) * in_channels
    # Compute the weight offset
    weight_offset = (block_idx * in_channels * kernel_size * kernel_size) + (out_h * kernel_size + out_w) * in_channels * kernel_size
    # Compute the output offset
    out_offset = (block_idx * out_channels * height * width) + (out_h * width + out_w) * out_channels
    # Load input values
    input_val = tl.load(input_ptr + in_offset, other=0.0)
    # Load weight values
    weight_val = tl.load(weight_ptr + weight_offset, other=0.0)
    # Perform the convolution
    output_val = input_val * weight_val
    # Store the result
    tl.store(output_ptr + out_offset, output_val)


def triton_depthwise_conv(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int, stride: int, padding: int):
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
    output = torch.empty(batch_size, out_channels, height, width, device=input.device, dtype=input.dtype)

    # Number of elements in the tensor
    n_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    depthwise_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float16))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(in_channels, device='cuda', dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        out_channels = self.in_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        # Perform the convolution
        output = triton_depthwise_conv(x, self.weight, self.bias, batch_size, self.in_channels, out_channels, height, width, kernel_size, stride, padding)
        return output