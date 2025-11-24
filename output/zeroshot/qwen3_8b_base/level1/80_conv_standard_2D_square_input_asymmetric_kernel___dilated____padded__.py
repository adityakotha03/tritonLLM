import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, height, width)
    weight_ptr,  # Pointer to weight tensor (out_channels, in_channels, kernel_h, kernel_w)
    output_ptr,  # Pointer to output tensor (batch, out_channels, output_h, output_w)
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
    # Get program ID (block index)
    pid = tl.program_id(0)
    # Get thread ID within the block
    tid = tl.program_id(1)
    # Each block processes a contiguous block of output elements
    output_idx = pid * BLOCK_SIZE + tid
    # Compute the corresponding input position
    # Output shape: (batch, out_channels, output_h, output_w)
    # Input shape: (batch, in_channels, input_h, input_w)
    # Output indices: (b, oc, oh, ow)
    # Input indices: (b, ic, ih, iw)
    # Compute the input indices for this output position
    b = output_idx // (out_channels * BLOCK_SIZE * BLOCK_SIZE)
    oc = (output_idx // (BLOCK_SIZE * BLOCK_SIZE)) % out_channels
    oh = output_idx // (BLOCK_SIZE * BLOCK_SIZE) % out_channels
    ow = output_idx % BLOCK_SIZE

    # Compute the corresponding input indices with padding and dilation
    ih_start = (oh * stride_h - padding_h) + (dilation_h - 1) * (oh // (kernel_h - 1))
    iw_start = (ow * stride_w - padding_w) + (dilation_w - 1) * (ow // (kernel_w - 1))
    # Iterate over the kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute the input index
            ih = ih_start + kh * dilation_h
            iw = iw_start + kw * dilation_w
            # Check if the input index is valid
            if ih < 0 or ih >= height or iw < 0 or iw >= width:
                continue
            # Compute the input offset
            in_offset = b * in_channels * height * width + \
                        (oc // out_channels) * height * width + \
                        ih * width + iw
            # Load input value
            input_val = tl.load(input_ptr + in_offset, mask=(ih >= 0) & (ih < height) & (iw >= 0) & (iw < width), other=0.0)
            # Load weight value
            weight_val = tl.load(weight_ptr + oc * in_channels * kernel_h * kernel_w + (oc // out_channels) * kernel_h * kernel_w + kh * kernel_w + kw, other=0.0)
            # Multiply and accumulate
            output_val = tl.load(output_ptr + b * out_channels * output_h * output_w + oc * output_h * output_w + oh * output_w + ow, other=0.0)
            output_val += input_val * weight_val
            tl.store(output_ptr + b * out_channels * output_h * output_w + oc * output_h * output_w + oh * output_w + ow, output_val)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1)):
    """
    Custom Triton implementation of 2D convolution.
    """
    # Ensure input and weight are on GPU
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA."
    # Compute output dimensions
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    output_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    output_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_h, output_w, device=input.device, dtype=input.dtype)
    # Prepare output tensor for Triton
    output = output.contiguous()
    # Set kernel parameters
    BLOCK_SIZE = 128  # Tunable parameter for block size
    # Determine grid size
    grid = (batch_size * out_channels * output_h * output_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    conv2d_kernel[grid, 128](input, weight, output, batch_size, in_channels, out_channels, kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], BLOCK_SIZE)
    # Add bias if present
    if bias is not None:
        output += bias.view(1, out_channels, 1, 1)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)