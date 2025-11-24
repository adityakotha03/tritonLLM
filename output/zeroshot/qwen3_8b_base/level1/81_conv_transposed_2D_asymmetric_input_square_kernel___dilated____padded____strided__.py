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
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the thread index
    pid = tl.program_id(0)
    # Compute the block index within the output
    block_idx = pid // (out_channels // BLOCK_SIZE)
    # Compute the channel index within the output
    channel_idx = pid % (out_channels // BLOCK_SIZE) * BLOCK_SIZE
    # Compute the output spatial dimensions
    out_h = (input_ptr.shape[2] - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    out_w = (input_ptr.shape[3] - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # Compute the output offset for this block
    out_offset = block_idx * out_channels * out_h * out_w
    # Compute the input offset for this block
    in_offset = 0
    # Iterate over the output spatial dimensions
    for h in range(out_h):
        for w in range(out_w):
            # Compute the input spatial dimensions
            in_h = h // stride
            in_w = w // stride
            # Compute the input offset for this spatial position
            in_offset = in_h * input_ptr.shape[3] + in_w
            # Compute the output offset for this spatial position
            out_offset = block_idx * out_channels * out_h * out_w + channel_idx + h * out_w + w
            # Load the input values
            input_val = tl.load(input_ptr + in_offset, mask=in_offset < input_ptr.shape[1] * input_ptr.shape[2] * input_ptr.shape[3], other=0.0)
            # Load the weight values
            weight_val = tl.load(weight_ptr + channel_idx + in_offset, mask=channel_idx + in_offset < weight_ptr.shape[0] * weight_ptr.shape[1] * weight_ptr.shape[2] * weight_ptr.shape[3], other=0.0)
            # Perform the convolution
            output_val = input_val * weight_val
            # Store the output value
            tl.store(output_ptr + out_offset, output_val, mask=out_offset < output_ptr.shape[0] * output_ptr.shape[1] * output_ptr.shape[2] * output_ptr.shape[3])


def triton_conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    This function wraps the Triton kernel call for transposed convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    output = torch.empty_like(input)
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = output.shape[1]
    kernel_size = weight.shape[2]
    stride = 5
    padding = 1
    dilation = 2
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((batch_size * out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
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
        Performs the 2D transposed convolution using a custom Triton kernel.
        """
        weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).cuda()
        if self.bias:
            bias = torch.randn(self.out_channels).cuda()
        else:
            bias = None
        return triton_conv_transpose2d(x, weight, bias)