import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_relu_add_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Kernel size
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes a single output position
    # We'll process each output position in a block
    # We'll use shared memory to store the input patch
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // width
    out_w = pid % width
    # Compute the input position
    input_h = out_h * kernel_size - (kernel_size // 2)
    input_w = out_w * kernel_size - (kernel_size // 2)
    # Check if input is within bounds
    if input_h < 0 or input_h >= height or input_w < 0 or input_w >= width:
        return
    # Compute the input offset
    input_offset = (input_h * width + input_w) * in_channels
    # Compute the output offset
    out_offset = (out_h * width + out_w) * out_channels
    # Load the weight
    weight = tl.load(weight_ptr + tl.arange(0, out_channels) * in_channels + tl.arange(0, in_channels), mask=tl.arange(0, out_channels) * in_channels + tl.arange(0, in_channels) < out_channels * in_channels)
    # Load the input patch
    input_patch = tl.zeros((kernel_size * kernel_size * in_channels), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            input_patch[i * kernel_size * in_channels + j * in_channels : (i + 1) * kernel_size * in_channels] = tl.load(x_ptr + (input_h + i) * width * in_channels + (input_w + j) * in_channels + tl.arange(0, in_channels), mask=tl.arange(0, in_channels) < in_channels)
    # Compute the convolution
    conv = tl.dot(input_patch, weight)
    # Apply ReLU
    conv = tl.maximum(conv, 0.0)
    # Add bias
    conv += tl.load(bias_ptr + out_h * width + out_w)
    # Store the result
    tl.store(out_ptr + out_offset, conv)


def triton_conv_relu_add(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, height, width), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = batch_size * height * width

    # Determine the number of blocks needed
    grid = lambda meta: (n_elements,)

    # Launch the Triton kernel
    conv_relu_add_kernel[grid](x, weight, bias, out, batch_size, in_channels, out_channels, height, width, kernel_size, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_shape = bias_shape

    def forward(self, x):
        # Perform convolution, ReLU, and bias addition using Triton kernel
        x = triton_conv_relu_add(
            x,
            torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).cuda(),
            torch.randn(self.out_channels, 1, 1).cuda(),
            x.size(0),
            self.in_channels,
            self.out_channels,
            x.size(2),
            x.size(3),
            self.kernel_size
        )
        return x