import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor
    stride_h, stride_w,  # Stride parameters
    kernel_h, kernel_w,  # Kernel size
    padding_h, padding_w,  # Padding parameters
    output_padding_h, output_padding_w,  # Output padding parameters
    batch_size, in_channels, out_channels, height, width,  # Input dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_offset = pid * BLOCK_SIZE
    # Compute the output position
    oh = block_offset // (width * out_channels)
    ow = (block_offset % (width * out_channels)) // out_channels
    oc = block_offset % out_channels

    # Compute the input position
    ih = oh * stride_h - padding_h
    iw = ow * stride_w - padding_w

    # Initialize the output value
    out_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute the input position with kernel offset
            ih_k = ih + kh
            iw_k = iw + kw
            # Check if the input position is valid
            if ih_k < 0 or iw_k < 0 or ih_k >= height or iw_k >= width:
                continue
            # Compute the input offset
            input_offset = (ih_k * width + iw_k) * in_channels + oc
            # Load the input value
            input_val = tl.load(input_ptr + input_offset, mask=(ih_k >= 0) & (iw_k >= 0) & (ih_k < height) & (iw_k < width), other=0.0)
            # Multiply by weight
            weight_offset = (kh * kernel_w + kw) * in_channels + oc * in_channels
            weight_val = tl.load(weight_ptr + weight_offset, other=0.0)
            out_val += input_val * weight_val

    # Add bias
    bias_val = tl.load(bias_ptr + oc, other=0.0)
    out_val += bias_val

    # Compute the final output position
    final_oh = oh + output_padding_h
    final_ow = ow + output_padding_w
    output_offset = (final_oh * width + final_ow) * out_channels + oc
    tl.store(output_ptr + output_offset, out_val)


@triton.jit
def min_kernel(
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
    # Compute the minimum
    min_val = tl.min(x, mask=mask)
    # Store the result
    tl.store(output_ptr + block_start, min_val, mask=mask)


@triton.jit
def sum_kernel(
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
    # Compute the sum
    sum_val = tl.sum(x, mask=mask)
    # Store the result
    tl.store(output_ptr + block_start, sum_val, mask=mask)


@triton.jit
def gelu_kernel(
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
    # Compute the GELU approximation
    x = x * tl.erf(x * 0.7978845608028654 / tl.sqrt(2.0)) + 0.5
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride_h, stride_w, kernel_h, kernel_w, padding_h, padding_w, output_padding_h, output_padding_w, batch_size, in_channels, out_channels, height, width):
    """
    This function wraps the Triton kernel call for convolution transpose.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out_channels_total = out_channels * height * width
    out = torch.empty((batch_size, out_channels, height + output_padding_h, width + output_padding_w), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_kernel[grid](x, weight, out, bias, stride_h, stride_w, kernel_h, kernel_w, padding_h, padding_w, output_padding_h, output_padding_w, batch_size, in_channels, out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_min(x: torch.Tensor, dim: int, keepdim: bool):
    """
    This function wraps the Triton kernel call for min operation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty((x.shape[0], x.shape[1], x.shape[2], 1), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    min_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_sum(x: torch.Tensor, dim: int, keepdim: bool):
    """
    This function wraps the Triton kernel call for sum operation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty((x.shape[0], x.shape[1], 1, 1), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sum_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_gelu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for GELU activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda())
        self.bias = nn.Parameter(torch.randn(*bias_shape).cuda())

    def forward(self, x):
        # Perform convolution transpose
        x = triton_conv_transpose(x, self.weight, self.bias, self.stride, self.stride, self.kernel_size, self.kernel_size, self.padding, self.padding, self.output_padding, self.output_padding, x.size(0), self.in_channels, self.out_channels, x.size(2), x.size(3))
        # Perform minimum operation along channel dimension
        x = triton_min(x, dim=1, keepdim=True)
        # Perform sum operation along height dimension
        x = triton_sum(x, dim=2, keepdim=True)
        # Apply GELU activation
        x = triton_gelu(x)
        # Add bias
        x = x + self.bias
        return x