import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transpose_conv_kernel(
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
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block index in the output
    block_idx = pid // (out_channels // BLOCK_SIZE)
    # Compute the channel index in the output
    channel_idx = pid % (out_channels // BLOCK_SIZE) * BLOCK_SIZE
    # Compute the offset in the output
    offset = tl.arange(0, BLOCK_SIZE)
    # Compute the output position
    output_h = block_idx * stride_h
    output_w = channel_idx + offset
    # Compute the input position
    input_h = (output_h - padding_h) // stride_h
    input_w = (output_w - padding_w) // stride_w
    # Compute the weight indices
    weight_h = tl.arange(0, kernel_h)
    weight_w = tl.arange(0, kernel_w)
    # Compute the input indices
    input_h_indices = input_h + weight_h
    input_w_indices = input_w + weight_w
    # Compute the output indices
    output_h_indices = output_h + weight_h
    output_w_indices = output_w + weight_w
    # Compute the input and weight pointers
    input_ptr = input_ptr + (input_h_indices * width + input_w_indices) * in_channels
    weight_ptr = weight_ptr + (weight_h * kernel_w + weight_w) * in_channels * groups
    # Compute the output pointer
    output_ptr = output_ptr + (output_h_indices * width + output_w_indices) * out_channels
    # Load input and weight
    input_val = tl.load(input_ptr, mask=input_h_indices < height, other=0.0)
    weight_val = tl.load(weight_ptr, mask=weight_h < kernel_h, other=0.0)
    # Compute the convolution
    output_val = tl.dot(input_val, weight_val)
    # Store the result
    tl.store(output_ptr, output_val, mask=output_h_indices < height + output_padding_h)


def triton_transpose_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: tuple, stride: int, padding: int, output_padding: int, groups: int):
    """
    This function wraps the Triton kernel call for transposed 2D convolution.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output shape
    height_out = (input.size(2) - 1) * stride + kernel_size[0] - 2 * padding + output_padding
    width_out = (input.size(3) - 1) * stride + kernel_size[1] - 2 * padding + output_padding
    output = torch.empty((batch_size, out_channels, height_out, width_out), device=input.device, dtype=input.dtype)

    # Number of elements in the output
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    transpose_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size[0], kernel_size[1], stride, stride, padding, padding, output_padding, output_padding, groups, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
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
        """
        # Create weight and bias tensors
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]))
        weight = weight.cuda()
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels))
            bias = bias.cuda()
        else:
            bias = None

        # Perform the transposed 2D convolution using the Triton kernel
        output = triton_transpose_conv2d(x, weight, bias, x.size(0), self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups)
        return output