import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1x1_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    output_shape,  # (N, C_out, H, W)
    stride,  # Convolution stride
    padding,  # Convolution padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output tensor
    n, c_out, h_out, w_out = output_shape
    n, c_in, h_in, w_in = input_shape

    # Compute the offset in the output tensor
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Compute the position in the output tensor
    h = offset // (w_out)
    w = offset % (w_out)
    # Compute the position in the input tensor
    h_in = h * stride - padding
    w_in = w * stride - padding
    # Compute the input channel index
    c_in_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the output channel index
    c_out_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the input and output indices
    input_indices = (n, c_in_idx, h_in, w_in)
    output_indices = (n, c_out_idx, h, w)
    # Load input and weights
    input_vals = tl.load(input_ptr + input_indices, mask=c_in_idx < c_in, other=0.0)
    weight_vals = tl.load(weight_ptr + output_indices, mask=c_out_idx < c_out, other=0.0)
    # Perform the convolution
    output_vals = tl.dot(input_vals, weight_vals)
    # Store the result
    tl.store(output_ptr + output_indices, output_vals, mask=c_out_idx < c_out)


def triton_conv1x1(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output_shape = (input.size(0), weight.size(0), input.size(2), input.size(3))
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1x1_kernel[grid](input, weight, output, input.shape, output.shape, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def conv3x3_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    output_shape,  # (N, C_out, H, W)
    stride,  # Convolution stride
    padding,  # Convolution padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output tensor
    n, c_out, h_out, w_out = output_shape
    n, c_in, h_in, w_in = input_shape

    # Compute the offset in the output tensor
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Compute the position in the output tensor
    h = offset // (w_out)
    w = offset % (w_out)
    # Compute the position in the input tensor
    h_in = h * stride - padding
    w_in = w * stride - padding
    # Compute the input channel index
    c_in_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the output channel index
    c_out_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the input and output indices
    input_indices = (n, c_in_idx, h_in, w_in)
    output_indices = (n, c_out_idx, h, w)
    # Load input and weights
    input_vals = tl.load(input_ptr + input_indices, mask=c_in_idx < c_in, other=0.0)
    weight_vals = tl.load(weight_ptr + output_indices, mask=c_out_idx < c_out, other=0.0)
    # Perform the convolution
    output_vals = tl.dot(input_vals, weight_vals)
    # Store the result
    tl.store(output_ptr + output_indices, output_vals, mask=c_out_idx < c_out)


def triton_conv3x3(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output_shape = (input.size(0), weight.size(0), input.size(2), input.size(3))
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3x3_kernel[grid](input, weight, output, input.shape, output.shape, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def conv5x5_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    output_shape,  # (N, C_out, H, W)
    stride,  # Convolution stride
    padding,  # Convolution padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output tensor
    n, c_out, h_out, w_out = output_shape
    n, c_in, h_in, w_in = input_shape

    # Compute the offset in the output tensor
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Compute the position in the output tensor
    h = offset // (w_out)
    w = offset % (w_out)
    # Compute the position in the input tensor
    h_in = h * stride - padding
    w_in = w * stride - padding
    # Compute the input channel index
    c_in_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the output channel index
    c_out_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the input and output indices
    input_indices = (n, c_in_idx, h_in, w_in)
    output_indices = (n, c_out_idx, h, w)
    # Load input and weights
    input_vals = tl.load(input_ptr + input_indices, mask=c_in_idx < c_in, other=0.0)
    weight_vals = tl.load(weight_ptr + output_indices, mask=c_out_idx < c_out, other=0.0)
    # Perform the convolution
    output_vals = tl.dot(input_vals, weight_vals)
    # Store the result
    tl.store(output_ptr + output_indices, output_vals, mask=c_out_idx < c_out)


def triton_conv5x5(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output_shape = (input.size(0), weight.size(0), input.size(2), input.size(3))
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv5x5_kernel[grid](input, weight, output, input.shape, output.shape, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def max_pool_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    output_shape,  # (N, C, H, W)
    kernel_size,  # Pooling kernel size
    stride,  # Pooling stride
    padding,  # Pooling padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output tensor
    n, c, h_out, w_out = output_shape
    n, c, h_in, w_in = input_shape

    # Compute the offset in the output tensor
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Compute the position in the output tensor
    h = offset // (w_out)
    w = offset % (w_out)
    # Compute the position in the input tensor
    h_in_start = h * stride - padding
    w_in_start = w * stride - padding
    # Compute the input indices
    input_indices = (n, tl.arange(0, BLOCK_SIZE), h_in_start, w_in_start)
    # Load input values
    input_vals = tl.load(input_ptr + input_indices, mask=tl.arange(0, BLOCK_SIZE) < c, other=-float('inf'))
    # Compute the max value
    max_val = tl.max(input_vals)
    # Store the result
    tl.store(output_ptr + (n, tl.arange(0, BLOCK_SIZE), h, w), max_val, mask=tl.arange(0, BLOCK_SIZE) < c)


def triton_max_pool(input: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Prepare output tensor
    output_shape = (input.size(0), input.size(1), input.size(2) // stride, input.size(3) // stride)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    max_pool_kernel[grid](input, output, input.shape, output.shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        self.branch1x1 = nn.Parameter(torch.randn(1, out_1x1, 1, 1).cuda())
        self.branch3x3 = nn.Parameter(torch.randn(1, out_3x3, 1, 1).cuda())
        self.branch3x3_weights = nn.Parameter(torch.randn(1, reduce_3x3, 1, 1).cuda())
        self.branch5x5 = nn.Parameter(torch.randn(1, out_5x5, 1, 1).cuda())
        self.branch5x5_weights = nn.Parameter(torch.randn(1, reduce_5x5, 1, 1).cuda())
        self.branch_pool = nn.Parameter(torch.randn(1, pool_proj, 1, 1).cuda())

    def forward(self, x):
        # 1x1 convolution branch
        branch1x1 = triton_conv1x1(x, self.branch1x1, stride=1, padding=0)
        # 3x3 convolution branch
        branch3x3 = triton_conv3x3(x, self.branch3x3_weights, stride=1, padding=1)
        branch3x3 = triton_conv3x3(branch3x3, self.branch3x3, stride=1, padding=0)
        # 5x5 convolution branch
        branch5x5 = triton_conv5x5(x, self.branch5x5_weights, stride=1, padding=2)
        branch5x5 = triton_conv5x5(branch5x5, self.branch5x5, stride=1, padding=0)
        # Max pooling branch
        branch_pool = triton_max_pool(x, kernel_size=3, stride=1, padding=1)
        branch_pool = triton_conv1x1(branch_pool, self.branch_pool, stride=1, padding=0)
        # Concatenate all branches
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)