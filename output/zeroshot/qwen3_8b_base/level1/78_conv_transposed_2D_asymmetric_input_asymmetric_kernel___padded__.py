import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    padding_h,  # Padding height
    padding_w,  # Padding width
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the output position
    out_h = pid // (out_channels // BLOCK_SIZE)
    out_w = pid % (out_channels // BLOCK_SIZE)
    # Compute the output index
    out_idx = out_h * out_channels + out_w
    # Compute the input position
    in_h = out_h * stride_h - padding_h
    in_w = out_w * stride_w - padding_w
    # Compute the input offset
    input_offset = (in_h * in_channels + in_w) * out_channels
    # Compute the weight offset
    weight_offset = out_idx * in_channels
    # Compute the output offset
    output_offset = out_idx * (in_channels * kernel_h * kernel_w)
    # Load input and weight
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, BLOCK_SIZE) < in_channels, other=0.0)
    weight_val = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < in_channels, other=0.0)
    # Compute the output
    output_val = tl.dot(input_val, weight_val)
    # Store the output
    tl.store(output_ptr + output_offset, output_val)


def triton_conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Prepare output tensor
    output_shape = (input.size(0), weight.size(0), input.size(2) * stride_h + 2 * padding_h, input.size(3) * stride_w + 2 * padding_w)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](input, weight, output, input.size(0), input.size(1), weight.size(0), kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_h, kernel_w))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = triton_conv_transpose2d(x, self.weight, self.bias if self.bias else None)
        return output