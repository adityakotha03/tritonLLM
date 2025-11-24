import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transpose_conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    output_padding_d: tl.constexpr,
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output
    pid = tl.program_id(0)
    # Compute the position in the output
    out_d = pid // (out_channels * kernel_h * kernel_w)
    out_h = (pid // (out_channels * kernel_w)) % kernel_h
    out_w = pid % kernel_w

    # Compute the corresponding input position
    input_d = out_d - (output_padding_d - padding_d)
    input_h = out_h - (output_padding_h - padding_h)
    input_w = out_w - (output_padding_w - padding_w)

    # Compute the output channel index
    oc = pid % out_channels
    # Compute the input channel index
    ic = (pid // out_channels) % in_channels
    # Compute the group index
    g = (pid // (out_channels * in_channels)) % groups

    # Compute the offset in the input
    input_offset = (input_d * height * width + input_h * width + input_w) * in_channels + ic
    # Compute the offset in the weight
    weight_offset = (oc * in_channels // groups * kernel_d * kernel_h * kernel_w + g * kernel_d * kernel_h * kernel_w + input_d * kernel_h * kernel_w + input_h * kernel_w + input_w)
    # Compute the offset in the output
    output_offset = (out_d * height * width + out_h * width + out_w) * out_channels + oc

    # Load input and weight
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, BLOCK_SIZE) < in_channels, other=0.0)
    weight_val = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < kernel_d, other=0.0)

    # Perform the convolution
    out_val = tl.dot(input_val, weight_val)

    # Store the result
    tl.store(output_ptr + output_offset, out_val)


def triton_transpose_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, output_padding: tuple, groups: int):
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
    output = torch.empty((batch_size, out_channels, input.size(2) + output_padding[0] + output_padding[1] + output_padding[2] - 1), device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    transpose_conv3d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2], stride[0], stride[1], stride[2], padding[0], padding[1], padding[2], output_padding[0], output_padding[1], output_padding[2], groups, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Compute output dimensions
        batch_size = x.size(0)
        depth = x.size(2)
        height = x.size(3)
        width = x.size(4)

        # Compute output dimensions
        out_depth = (depth - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        out_height = (height - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        out_width = (width - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]

        # Initialize weight and bias
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        bias = torch.nn.Parameter(torch.randn(self.out_channels)) if self.bias else None

        # Perform the transposed convolution
        output = triton_transpose_conv3d(x, weight, bias, batch_size, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups)
        return output