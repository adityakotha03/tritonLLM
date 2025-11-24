import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // (BLOCK_SIZE // stride_h)
    out_w = pid % (BLOCK_SIZE // stride_h)
    # Compute the input position
    in_h = out_h * stride_h - padding_h
    in_w = out_w * stride_w - padding_w
    # Compute the start and end indices for the kernel
    start_h = tl.max(tl.arange(0, kernel_h) * dilation_h - padding_h, 0)
    start_w = tl.max(tl.arange(0, kernel_w) * dilation_w - padding_w, 0)
    end_h = start_h + kernel_h
    end_w = start_w + kernel_w
    # Compute the input and weight indices
    input_idx = (tl.arange(0, BLOCK_SIZE // stride_h) * stride_h + tl.arange(0, BLOCK_SIZE // stride_w) * stride_w) + tl.arange(0, kernel_h) * dilation_h + tl.arange(0, kernel_w) * dilation_w
    input_idx = input_idx + (tl.arange(0, in_channels) // groups) * (in_channels // groups) * (height * width) + tl.arange(0, in_channels) // groups * (width) + tl.arange(0, width) + tl.arange(0, height) * width
    input_idx = input_idx + (out_h * stride_h - padding_h) * width + (out_w * stride_w - padding_w)
    weight_idx = (tl.arange(0, kernel_h) * kernel_w + tl.arange(0, kernel_w)) + (tl.arange(0, out_channels) // groups) * (out_channels // groups) * (kernel_h * kernel_w) + tl.arange(0, out_channels) // groups * kernel_w + tl.arange(0, kernel_w)
    weight_idx = weight_idx + (tl.arange(0, in_channels) // groups) * (in_channels // groups) * (kernel_h * kernel_w) + tl.arange(0, in_channels) // groups * kernel_w + tl.arange(0, kernel_w)
    # Load input and weight
    input_val = tl.load(input_ptr + input_idx, mask=tl.arange(0, kernel_h * kernel_w) < kernel_h * kernel_w, other=0.0)
    weight_val = tl.load(weight_ptr + weight_idx, mask=tl.arange(0, kernel_h * kernel_w) < kernel_h * kernel_w, other=0.0)
    # Perform the convolution
    output_val = tl.dot(input_val, weight_val)
    # Store the result
    tl.store(output_ptr + (out_h * width + out_w), output_val, mask=tl.arange(0, out_channels) < out_channels)


def triton_transposed_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int):
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
    output = torch.zeros(batch_size, out_channels, input.size(2) * stride[0] - 2 * padding[0], input.size(3) * stride[1] - 2 * padding[1], device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    transposed_conv2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
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
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Initialize weight and bias
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]))
        bias = torch.nn.Parameter(torch.randn(self.out_channels)) if self.bias else None

        # Perform the transposed convolution using the custom Triton kernel
        output = triton_transposed_conv2d(x, weight, bias, x.size(0), self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        return output