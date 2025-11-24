import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_channels,  # Input channels
    out_channels,  # Output channels
    kernel_size,  # Kernel size (3D)
    stride,  # Stride (3D)
    padding,  # Padding (3D)
    output_padding,  # Output padding (3D)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_id = pid // (out_channels // BLOCK_SIZE)
    channel_id = pid % (out_channels // BLOCK_SIZE) * BLOCK_SIZE

    # Compute the output dimensions
    D_out = (D + 2 * padding - kernel_size + output_padding) // stride + 1
    H_out = (H + 2 * padding - kernel_size + output_padding) // stride + 1
    W_out = (W + 2 * padding - kernel_size + output_padding) // stride + 1

    # Compute the output index for this block
    out_idx = block_id * (D_out * H_out * W_out) + channel_id
    offset = out_idx + tl.arange(0, BLOCK_SIZE)
    mask = offset < (D_out * H_out * W_out * out_channels)

    # Compute the input and weight indices
    input_offset = tl.load(input_ptr + offset, mask=mask, other=0.0)
    weight_offset = tl.load(weight_ptr + offset, mask=mask, other=0.0)

    # Perform the convolution operation
    output = tl.dot(input_offset, weight_offset)

    # Store the result
    tl.store(output_ptr + offset, output, mask=mask)


def triton_conv_transpose3d(input, weight, batch_size, in_channels, out_channels, kernel_size, stride, padding, output_padding):
    """
    Custom Triton implementation of 3D transposed convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty((batch_size, out_channels, D_out, H_out, W_out), device=input.device)

    # Determine block size and grid size
    BLOCK_SIZE = 128
    grid = lambda meta: (out_channels // BLOCK_SIZE + 1,)

    # Launch the Triton kernel
    conv_transpose3d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_id = pid // (n_elements // BLOCK_SIZE)
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    input = tl.load(input_ptr + offset, mask=mask, other=0.0)

    # Compute softmax
    max_val = tl.max(input, axis=0)
    exp_input = tl.exp(input - max_val)
    sum_exp = tl.sum(exp_input, axis=0)
    output = exp_input / sum_exp

    # Store the result
    tl.store(output_ptr + offset, output, mask=mask)


def triton_softmax(input, dim):
    """
    Custom Triton implementation of softmax.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    # Determine block size and grid size
    BLOCK_SIZE = 128
    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softmax_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def sigmoid_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_id = pid // (n_elements // BLOCK_SIZE)
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    input = tl.load(input_ptr + offset, mask=mask, other=0.0)

    # Compute sigmoid
    output = 1.0 / (1.0 + tl.exp(-input))

    # Store the result
    tl.store(output_ptr + offset, output, mask=mask)


def triton_sigmoid(input):
    """
    Custom Triton implementation of sigmoid.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    # Determine block size and grid size
    BLOCK_SIZE = 128
    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sigmoid_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for 3D transposed convolution, softmax, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        # Custom Triton-based 3D transposed convolution
        x = triton_conv_transpose3d(x, self.weight, x.size(0), self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding)
        # Custom Triton-based softmax
        x = triton_softmax(x, dim=1)
        # Custom Triton-based sigmoid
        x = triton_sigmoid(x)
        return x