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
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles a single output element
    pid = tl.program_id(0)
    # Compute the output position (batch, out_channel, out_h, out_w)
    out_h = pid // (width * out_channels)
    out_w = (pid // out_channels) % width
    out_c = pid % out_channels

    # Compute the input position (batch, in_channel, in_h, in_w)
    batch_id = pid // (width * out_channels * out_channels)
    in_c = out_c  # Depthwise: in_channels == out_channels
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding

    # Iterate over the kernel
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute the input position for this kernel element
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            # Check if the input position is valid
            if in_h < 0 or in_h >= height or in_w < 0 or in_w >= width:
                continue
            # Compute the input index
            in_idx = batch_id * in_channels * height * width + in_c * height * width + in_h * width + in_w
            # Load input value
            input_val = tl.load(input_ptr + in_idx, mask=tl.arange(0, 1) < 1, other=0.0)
            # Load weight value
            weight_idx = out_c * in_channels * kernel_size * kernel_size + in_c * kernel_size * kernel_size + kh * kernel_size + kw
            weight_val = tl.load(weight_ptr + weight_idx, mask=tl.arange(0, 1) < 1, other=0.0)
            # Multiply and accumulate
            output_val = tl.load(output_ptr + pid, mask=tl.arange(0, 1) < 1, other=0.0)
            output_val += input_val * weight_val
            tl.store(output_ptr + pid, output_val, mask=tl.arange(0, 1) < 1)

def triton_depthwise_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """
    Perform a depthwise 2D convolution using a Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)

    batch_size = input.size(0)
    in_channels = input.size(1)
    out_channels = weight.size(0)
    height = input.size(2)
    width = input.size(3)

    # Compute output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, device=input.device, dtype=input.dtype)

    # Determine the number of blocks needed
    num_blocks = batch_size * out_channels * out_h * out_w
    BLOCK_SIZE = 1

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    depthwise_conv2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using a Triton kernel.
        """
        # Create weight tensor
        weight = torch.empty(self.in_channels, self.in_channels, self.kernel_size, 1, device=x.device, dtype=x.dtype)
        # Initialize weights with random values
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        # Create bias tensor if needed
        bias = None
        if self.bias:
            bias = torch.empty(self.in_channels, device=x.device, dtype=x.dtype)
            torch.nn.init.constant_(bias, 0)
        # Perform convolution using Triton kernel
        return triton_depthwise_conv2d(x, weight, bias, self.stride, self.padding, self.dilation)