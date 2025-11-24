import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in, H, W)
    weight_shape,  # (C_out, C_in, K, K)
    output_shape,  # (N, C_out, H_out, W_out)
    stride,  # Stride of the convolution
    padding,  # Padding of the convolution
    output_padding,  # Output padding
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of output elements
    pid = tl.program_id(0)
    # Calculate the output position (N, C_out, H_out, W_out)
    n = pid // (output_shape[1] * output_shape[2] * output_shape[3])
    c_out = (pid // (output_shape[2] * output_shape[3])) % output_shape[1]
    h_out = (pid // output_shape[3]) % output_shape[2]
    w_out = pid % output_shape[3]

    # Calculate the corresponding input position (N, C_in, H_in, W_in)
    # Output shape: (N, C_out, H_out, W_out)
    # Input shape: (N, C_in, H_in, W_in)
    # H_in = (H_out - 1) * stride - 2 * padding + kernel_size
    # W_in = (W_out - 1) * stride - 2 * padding + kernel_size
    # We assume input is padded appropriately
    h_in = (h_out - 1) * stride - 2 * padding + kernel_size
    w_in = (w_out - 1) * stride - 2 * padding + kernel_size

    # Calculate the starting index in input for this output position
    input_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + \
                c_in * input_shape[2] * input_shape[3] + \
                h_in * input_shape[3] + w_in

    # Calculate the starting index in weight for this output position
    weight_idx = c_out * weight_shape[1] * weight_shape[2] * weight_shape[3] + \
                 c_in * weight_shape[2] * weight_shape[3] + \
                 (kernel_size - 1) * weight_shape[3] + (kernel_size - 1)

    # Calculate the output index
    output_idx = n * output_shape[1] * output_shape[2] * output_shape[3] + \
                 c_out * output_shape[2] * output_shape[3] + \
                 h_out * output_shape[3] + w_out

    # Load input value
    input_val = tl.load(input_ptr + input_idx, other=0.0)
    # Load weight value
    weight_val = tl.load(weight_ptr + weight_idx, other=0.0)
    # Compute the product
    product = input_val * weight_val
    # Add bias
    bias_val = tl.load(bias_ptr + c_out, other=0.0)
    output_val = product + bias_val
    # Store output value
    tl.store(output_ptr + output_idx, output_val)


@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute softmax
    max_x = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp
    # Store the result
    tl.store(output_ptr + offsets, softmax, mask=mask)


@triton.jit
def sigmoid_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute sigmoid
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    # Store the result
    tl.store(output_ptr + offsets, sigmoid, mask=mask)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride, padding, output_padding):
    """
    Custom Triton kernel for transposed convolution.
    """
    # Ensure inputs are on GPU and contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    output_shape = (x.size(0), weight.size(0), x.size(2) + 2 * padding - output_padding, x.size(3) + 2 * padding - output_padding)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    # Calculate the number of elements
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_kernel[grid](x, weight, bias, output, x.size(), weight.size(), output.size(), stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_softmax(x: torch.Tensor):
    """
    Custom Triton kernel for softmax.
    """
    # Ensure input is on GPU and contiguous
    x = x.contiguous()

    # Prepare output tensor
    output = torch.empty_like(x)

    # Calculate the number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softmax_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_sigmoid(x: torch.Tensor):
    """
    Custom Triton kernel for sigmoid.
    """
    # Ensure input is on GPU and contiguous
    x = x.contiguous()

    # Prepare output tensor
    output = torch.empty_like(x)

    # Calculate the number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for transposed convolution, softmax, bias addition, scaling, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        self.scaling_factor = scaling_factor

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(*bias_shape))

    def forward(self, x):
        # Custom transposed convolution
        x = triton_conv_transpose(x, self.weight, self.bias, self.stride, self.padding, self.output_padding)
        # Custom softmax
        x = triton_softmax(x)
        # Bias addition
        x = x + self.bias
        # Scaling
        x = x * self.scaling_factor
        # Custom sigmoid
        x = triton_sigmoid(x)
        return x