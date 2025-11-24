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
    height_in: tl.constexpr,
    width_in: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the output position
    oh = pid // width_in
    ow = pid % width_in

    # Compute the input position
    # Output shape: (batch, out_channels, H, W)
    # Input shape: (batch, in_channels, H, W)
    # Weight shape: (out_channels, in_channels // groups, kernel_h, kernel_w)
    # Output is computed as sum over (k_h, k_w) of input[batch, g * (out_ch // groups) + i, oh - k_h * dilation_h + ...]
    # We need to compute the input positions for each kernel position

    # For each output position (oh, ow), compute the input positions
    # Input is padded with padding_h and padding_w
    # Output is padded with output_padding_h and output_padding_w
    # The formula for input positions is:
    # input_h = oh * stride_h - (kernel_h - 1) * dilation_h + padding_h
    # input_w = ow * stride_w - (kernel_w - 1) * dilation_w + padding_w
    # We need to iterate over all kernel positions (kh, kw)
    # For each kernel position, compute the input position and accumulate

    # We will use a loop over the kernel positions
    # To optimize, we can compute the input positions for all kernel positions
    # and then compute the sum over them

    # Compute the input positions for all kernel positions
    # We will use a loop over the kernel positions
    # Each thread handles one kernel position
    # We will use a loop over the kernel positions

    # We will compute the input positions for each kernel position
    # and then accumulate the result

    # Initialize output
    out = tl.zeros((out_channels,), dtype=tl.float32)

    # Iterate over kernel positions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute the input position for this kernel position
            input_h = oh * stride_h - (kh) * dilation_h + padding_h
            input_w = ow * stride_w - (kw) * dilation_w + padding_w

            # Check if input_h and input_w are within the input range
            if input_h < 0 or input_h >= height_in or input_w < 0 or input_w >= width_in:
                continue

            # Compute the input index
            input_idx = (input_h * width_in + input_w) * in_channels + pid % groups * (in_channels // groups)
            input_val = tl.load(input_ptr + input_idx, mask=input_idx < (height_in * width_in * in_channels), other=0.0)

            # Compute the weight index
            weight_idx = (kh * kernel_w + kw) * (in_channels // groups) + pid % groups * (out_channels // groups)
            weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < (kernel_h * kernel_w * (in_channels // groups) * out_channels), other=0.0)

            # Multiply and accumulate
            out += input_val * weight_val

    # Store the result
    output_idx = (oh * width_in + ow) * out_channels + pid % groups * (out_channels // groups)
    tl.store(output_ptr + output_idx, out, mask=output_idx < (height_in * width_in * out_channels), other=0.0)


def triton_conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, output_padding: tuple, dilation: tuple, groups: int):
    """
    This function wraps the Triton kernel call for transposed 2D convolution.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Compute output dimensions
    height_out = (input.size(2) - 1) * stride[0] + kernel_size[0] - 2 * padding[0] + output_padding[0]
    width_out = (input.size(3) - 1) * stride[1] + kernel_size[1] - 2 * padding[1] + output_padding[1]

    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, dtype=input.dtype, device=input.device)

    # Number of elements in the output
    n_elements = output.numel()

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, input.size(2), input.size(3), kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1], output_padding[0], output_padding[1], dilation[0], dilation[1], groups, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom Triton kernels.
        """
        # Perform the transposed 2D convolution
        output = triton_conv_transpose2d(
            x, self.weight, self.bias, x.size(0), self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding, self.output_padding, self.dilation, self.groups
        )
        return output