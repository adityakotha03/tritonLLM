import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
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
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset in the output
    block_idx = pid * BLOCK_SIZE
    # Compute the output coordinates
    oh = block_idx // (width // stride_w)
    ow = block_idx % (width // stride_w)
    # Compute the input coordinates
    ih = oh * stride_h - padding_h
    iw = ow * stride_w - padding_w
    # Compute the number of elements in the block
    num_elements = (kernel_h * kernel_w * in_channels)
    # Compute the offset in the input and output
    input_offset = (tl.arange(0, BLOCK_SIZE) // (kernel_w * in_channels)) * (kernel_w * in_channels) + (tl.arange(0, BLOCK_SIZE) % (kernel_w * in_channels)) // in_channels * kernel_w + (tl.arange(0, BLOCK_SIZE) % (kernel_w * in_channels)) % in_channels
    output_offset = (oh * width + ow) * out_channels + tl.arange(0, BLOCK_SIZE) % out_channels
    # Load input values
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset < input_ptr.size, other=0.0)
    # Load weights
    weight_vals = tl.load(weight_ptr + tl.arange(0, num_elements), mask=tl.arange(0, num_elements) < weight_ptr.size, other=0.0)
    # Compute the convolution
    output_vals = tl.dot(input_vals, weight_vals)
    # Store output values
    tl.store(output_ptr + output_offset, output_vals, mask=output_offset < output_ptr.size)


def triton_depthwise_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1):
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
    output = torch.empty_like(input)
    # Compute the number of elements in the output
    out_channels = weight.size(0)
    width = input.size(3)
    height = input.size(2)
    # Compute the number of blocks needed
    num_blocks = (width // stride_w) * (height // stride_h)
    BLOCK_SIZE = 128  # Tunable parameter for block size
    # Determine the number of blocks needed
    grid = lambda meta: (num_blocks,)
    # Launch the Triton kernel
    depthwise_conv2d_kernel[grid](input, weight, output, input.size(0), input.size(1), out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = triton_depthwise_conv2d(x, self.weight, self.bias, self.stride_h, self.stride_w, self.padding_h, self.padding_w, self.dilation_h, self.dilation_w)
        return output