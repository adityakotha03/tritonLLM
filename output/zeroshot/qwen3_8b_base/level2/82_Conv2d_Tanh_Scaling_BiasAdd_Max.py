import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # Kernel size (same for height and width)
    stride,  # Stride for convolution
    padding,  # Padding for convolution
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Compute the output position
    # Output shape: (N, C_out, H_out, W_out)
    # H_out = (H + 2*padding - kernel_size) // stride + 1
    # W_out = (W + 2*padding - kernel_size) // stride + 1
    # For simplicity, assume input is (1, C, H, W) and output is (1, C_out, H_out, W_out)
    # We'll process one output element per thread
    # Each thread computes one output element
    # We'll use a naive approach for demonstration; for performance, more sophisticated tiling is needed

    # Compute output coordinates
    out_idx = pid
    # Convert to output (N, C_out, H_out, W_out)
    n_out = input_shape[0]
    c_out = input_shape[1]
    h_out = input_shape[2]
    w_out = input_shape[3]
    out_n = out_idx // (c_out * h_out * w_out)
    out_c = (out_idx // (h_out * w_out)) % c_out
    out_h = (out_idx // w_out) % h_out
    out_w = out_idx % w_out

    # Compute input coordinates
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    in_h_end = in_h_start + kernel_size
    in_w_end = in_w_start + kernel_size

    # Initialize output
    out_val = 0.0

    # Iterate over kernel
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            in_h = in_h_start + k_h
            in_w = in_w_start + k_w
            if in_h < 0 or in_h >= input_shape[2] or in_w < 0 or in_w >= input_shape[3]:
                continue
            # Compute input index
            in_idx = out_n * input_shape[1] * input_shape[2] * input_shape[3] + out_c * input_shape[2] * input_shape[3] + in_h * input_shape[3] + in_w
            in_val = tl.load(input_ptr + in_idx, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=0.0)
            # Compute weight index
            weight_idx = out_c * input_shape[1] * kernel_size * kernel_size + in_c * kernel_size * kernel_size + k_h * kernel_size + k_w
            weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < out_channels * in_channels * kernel_size * kernel_size, other=0.0)
            out_val += in_val * weight_val

    # Store the result
    output_idx = out_n * c_out * h_out * w_out + out_c * h_out * w_out + out_h * w_out + out_w
    tl.store(output_ptr + output_idx, out_val)


@triton.jit
def tanh_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh
    out = tl.tanh(x)
    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


@triton.jit
def scale_add_kernel(
    input_ptr,  # Pointer to input tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    scaling_factor,  # Scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Load bias values
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    # Perform scaling and addition
    out = x * scaling_factor + b
    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


@triton.jit
def max_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    pool_kernel_size,  # Pool kernel size (same for height and width)
    stride,  # Stride for pooling
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Compute the output position
    # Output shape: (N, C, H_out, W_out)
    # H_out = (H + 2*padding - pool_kernel_size) // stride + 1
    # W_out = (W + 2*padding - pool_kernel_size) // stride + 1
    # For simplicity, assume input is (1, C, H, W) and output is (1, C, H_out, W_out)
    # We'll process one output element per thread
    # Each thread computes one output element

    # Compute output coordinates
    out_idx = pid
    # Convert to output (N, C, H_out, W_out)
    n_out = input_shape[0]
    c_out = input_shape[1]
    h_out = input_shape[2]
    w_out = input_shape[3]
    out_n = out_idx // (c_out * h_out * w_out)
    out_c = (out_idx // (h_out * w_out)) % c_out
    out_h = (out_idx // w_out) % h_out
    out_w = out_idx % w_out

    # Compute input coordinates
    in_h_start = out_h * stride
    in_w_start = out_w * stride
    in_h_end = in_h_start + pool_kernel_size
    in_w_end = in_w_start + pool_kernel_size

    # Initialize output
    max_val = -float('inf')

    # Iterate over kernel
    for k_h in range(pool_kernel_size):
        for k_w in range(pool_kernel_size):
            in_h = in_h_start + k_h
            in_w = in_w_start + k_w
            if in_h < 0 or in_h >= input_shape[2] or in_w < 0 or in_w >= input_shape[3]:
                continue
            # Compute input index
            in_idx = out_n * input_shape[1] * input_shape[2] * input_shape[3] + out_c * input_shape[2] * input_shape[3] + in_h * input_shape[3] + in_w
            in_val = tl.load(input_ptr + in_idx, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=-float('inf'))
            if in_val > max_val:
                max_val = in_val

    # Store the result
    output_idx = out_n * c_out * h_out * w_out + out_c * h_out * w_out + out_h * w_out + out_w
    tl.store(output_ptr + output_idx, max_val)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call for convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output_shape = (input.shape[0], weight.shape[0], (input.shape[2] + 2 * padding - weight.shape[2]) // stride + 1, (input.shape[3] + 2 * padding - weight.shape[2]) // stride + 1)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input.shape, weight.shape[2], stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_tanh(input: torch.Tensor):
    """
    This function wraps the Triton kernel call for tanh.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    tanh_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_scale_add(input: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    This function wraps the Triton kernel call for scaling and addition.
    """
    assert input.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    bias = bias.contiguous()
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    scale_add_kernel[grid](input, bias, output, n_elements, scaling_factor, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_max_pool2d(input: torch.Tensor, pool_kernel_size: int, stride: int):
    """
    This function wraps the Triton kernel call for max pooling.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output_shape = (input.shape[0], input.shape[1], (input.shape[2] + 2 * 0 - pool_kernel_size) // stride + 1, (input.shape[3] + 2 * 0 - pool_kernel_size) // stride + 1)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    max_pool2d_kernel[grid](input, output, input.shape, pool_kernel_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.bias_shape = bias_shape
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Convolution
        x = triton_conv2d(x, torch.randn((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).cuda(), stride=1, padding=1)
        # Tanh activation
        x = triton_tanh(x)
        # Scaling
        x = triton_scale_add(x, torch.randn(self.bias_shape).cuda(), self.scaling_factor)
        # Max-pooling
        x = triton_max_pool2d(x, self.pool_kernel_size, stride=2)
        return x