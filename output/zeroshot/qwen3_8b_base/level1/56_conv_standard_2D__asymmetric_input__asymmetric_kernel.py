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
    # Compute the position in the output tensor
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Compute the output position
    oh = offset // (out_channels * width)
    ow = (offset % (out_channels * width)) // out_channels
    oc = offset % out_channels
    # Compute the input position
    ih = oh * stride_h - padding_h
    iw = ow * stride_w - padding_w
    # Iterate over the output channel
    for oc in range(out_channels):
        # Iterate over the output height
        for oh in range(oh_max):
            # Iterate over the output width
            for ow in range(ow_max):
                # Compute the input position
                ih = oh * stride_h - padding_h
                iw = ow * stride_w - padding_w
                # Compute the input channel
                for ic in range(in_channels // groups):
                    # Compute the weight position
                    weight_h = ic * out_channels // groups
                    weight_w = oc * in_channels // groups
                    # Compute the input offset
                    input_offset = (batch_size * in_channels * (ih + dilation_h * (kernel_h - 1)) * width + ic * width + (iw + dilation_w * (kernel_w - 1)))
                    # Load input
                    input_val = tl.load(input_ptr + input_offset, mask=..., other=0.0)
                    # Load weight
                    weight_val = tl.load(weight_ptr + weight_offset, mask=..., other=0.0)
                    # Compute the output
                    output_val = input_val * weight_val
                    # Accumulate the output
                    output_val = tl.atomic_add(output_ptr + output_offset, output_val)
    return


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
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
    batch_size, in_channels, height, width = input.shape
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    out_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], BLOCK_SIZE=BLOCK_SIZE)
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
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        weight = torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]).cuda()
        bias = torch.randn(self.out_channels).cuda() if self.bias else None
        return triton_conv2d(x, weight, bias, self.stride, self.padding, self.dilation)