import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in, H, W)
    output_shape,  # (N, C_out, H_out, W_out)
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    output_padding,  # Output padding
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Get the block index
    block_idx = pid
    # Compute the output position
    out_h = block_idx // output_shape[2]
    out_w = block_idx % output_shape[2]
    # Compute the input position
    in_h = (out_h * stride) - padding
    in_w = (out_w * stride) - padding
    # Compute the output channel
    out_c = pid % output_shape[1]
    # Compute the input channel
    in_c = pid // output_shape[1]
    # Compute the output offset
    out_offset = out_c * output_shape[2] * output_shape[3] + out_h * output_shape[3] + out_w
    # Compute the input offset
    in_offset = in_c * input_shape[2] * input_shape[3] + in_h * input_shape[3] + in_w
    # Load input value
    x = tl.load(input_ptr + in_offset, other=0.0)
    # Load weight value
    w = tl.load(weight_ptr + out_c * output_shape[1] + in_c, other=0.0)
    # Compute output value
    out = x * w
    # Store output value
    tl.store(output_ptr + out_offset, out)


@triton.jit
def gavg_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_channels,  # Number of channels
    height,  # Height of input
    width,  # Width of input
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Get the block index
    block_idx = pid
    # Compute the channel index
    c = block_idx % n_channels
    # Compute the height index
    h = block_idx // n_channels
    # Compute the offset
    offset = c * height * width + h
    # Load input value
    x = tl.load(input_ptr + offset, other=0.0)
    # Compute the average
    avg = x / (height * width)
    # Store output value
    tl.store(output_ptr + c, avg)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, input_shape, output_shape, kernel_size, stride, padding, output_padding):
    """
    Triton implementation of transposed convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_kernel[grid](x, weight, output, input_shape, output_shape, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_gavg(x: torch.Tensor, n_channels, height, width):
    """
    Triton implementation of global average pooling.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    output = torch.empty(n_channels, device=x.device, dtype=x.dtype)
    n_elements = n_channels
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gavg_kernel[grid](x, output, n_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier

    def forward(self, x):
        # Compute output shape for transposed convolution
        batch_size = x.size(0)
        input_shape = (batch_size, self.in_channels, x.size(2), x.size(3))
        output_shape = (batch_size, self.out_channels, x.size(2) * self.stride, x.size(3) * self.stride)
        # Transposed convolution with Triton kernel
        x = triton_conv_transpose(x, torch.randn(self.out_channels, self.in_channels, device=x.device), input_shape, output_shape,
                                 self.kernel_size, self.stride, self.padding, self.output_padding)
        # Multiply by scalar
        x = x * self.multiplier
        # First global average pooling with Triton kernel
        x = triton_gavg(x, self.out_channels, x.size(2), x.size(3))
        # Second global average pooling with Triton kernel
        x = triton_gavg(x, self.out_channels, x.size(2), x.size(3))
        return x