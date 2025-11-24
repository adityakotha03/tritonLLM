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
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Get the thread index within the block
    thread_id = tl.program_id(1)
    # Compute the block dimensions
    block_h = (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    block_w = (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    # Compute the output dimensions
    output_h = (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    output_w = (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    # Compute the output index
    output_idx = pid * block_w + thread_id
    # Compute the input indices
    h = (output_idx // output_w) * stride[0] + tl.arange(0, kernel_size[0])
    w = (output_idx % output_w) * stride[1] + tl.arange(0, kernel_size[1])
    # Compute the input offset
    input_offset = tl.arange(0, kernel_size[0]) * input_shape[3] + tl.arange(0, kernel_size[1])
    # Compute the weight offset
    weight_offset = tl.arange(0, kernel_size[0]) * weight_shape[1] + tl.arange(0, kernel_size[1])
    # Compute the output offset
    output_offset = output_idx * input_shape[1] + tl.arange(0, input_shape[1])
    # Load input and weight
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, kernel_size[0]) < input_shape[2], other=0.0)
    weight_val = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, kernel_size[0]) < weight_shape[0], other=0.0)
    # Compute the dot product
    dot = tl.dot(input_val, weight_val)
    # Store the result
    tl.store(output_ptr + output_offset, dot, mask=tl.arange(0, input_shape[1]) < input_shape[1])


@triton.jit
def group_norm_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    gamma_ptr,  # Pointer to gamma tensor
    beta_ptr,  # Pointer to beta tensor
    num_groups,  # Number of groups
    group_size,  # Size of each group
    eps,  # Small value to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Get the thread index within the block
    thread_id = tl.program_id(1)
    # Compute the group index
    group_idx = pid // group_size
    # Compute the channel index
    channel_idx = pid % group_size
    # Compute the input offset
    input_offset = group_idx * group_size * input_shape[2] * input_shape[3] + channel_idx * input_shape[2] * input_shape[3]
    # Compute the output offset
    output_offset = group_idx * group_size * input_shape[2] * input_shape[3] + channel_idx * input_shape[2] * input_shape[3]
    # Compute the gamma and beta offset
    gamma_offset = group_idx * group_size + channel_idx
    beta_offset = gamma_offset
    # Load input values
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, input_shape[2]) < input_shape[2], other=0.0)
    # Compute mean and variance
    mean = tl.mean(input_val)
    var = tl.var(input_val)
    # Normalize
    normalized = (input_val - mean) / tl.sqrt(var + eps)
    # Scale and shift
    scaled = normalized * tl.load(gamma_ptr + gamma_offset)
    shifted = scaled + tl.load(beta_ptr + beta_offset)
    # Store the result
    tl.store(output_ptr + output_offset, shifted, mask=tl.arange(0, input_shape[2]) < input_shape[2])


@triton.jit
def maxpool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Get the thread index within the block
    thread_id = tl.program_id(1)
    # Compute the output dimensions
    output_h = (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    output_w = (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    # Compute the output index
    output_idx = pid * output_w + thread_id
    # Compute the input indices
    h = (output_idx // output_w) * stride[0] + tl.arange(0, kernel_size[0])
    w = (output_idx % output_w) * stride[1] + tl.arange(0, kernel_size[1])
    # Compute the input offset
    input_offset = tl.arange(0, kernel_size[0]) * input_shape[3] + tl.arange(0, kernel_size[1])
    # Load input values
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, kernel_size[0]) < input_shape[2], other=-float('inf'))
    # Compute the max
    max_val = tl.max(input_val)
    # Store the result
    tl.store(output_ptr + output_idx, max_val, mask=tl.arange(0, kernel_size[0]) < input_shape[2])


@triton.jit
def clamp_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    clamp_min,  # Minimum value
    clamp_max,  # Maximum value
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Get the thread index within the block
    thread_id = tl.program_id(1)
    # Compute the input index
    input_idx = pid * input_shape[2] * input_shape[3] + thread_id
    # Load input value
    input_val = tl.load(input_ptr + input_idx, mask=tl.arange(0, input_shape[2]) < input_shape[2], other=0.0)
    # Clamp the value
    clamped_val = tl.where(input_val < clamp_min, clamp_min, tl.where(input_val > clamp_max, clamp_max, input_val))
    # Store the result
    tl.store(output_ptr + input_idx, clamped_val, mask=tl.arange(0, input_shape[2]) < input_shape[2])


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, kernel_size, stride, padding):
    input_shape = input.shape
    weight_shape = weight.shape
    output_shape = (
        input_shape[0],
        weight_shape[0],
        (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
        (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    )
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: ((output_shape[2] * output_shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    conv2d_kernel[grid](input, weight, output, input_shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_group_norm(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups, eps):
    input_shape = input.shape
    group_size = input_shape[1] // num_groups
    output_shape = input_shape
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: ((input_shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    group_norm_kernel[grid](input, output, gamma, beta, num_groups, group_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_maxpool2d(input: torch.Tensor, kernel_size, stride, padding):
    input_shape = input.shape
    output_shape = (
        input_shape[0],
        input_shape[1],
        (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
        (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    )
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: ((output_shape[2] * output_shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    maxpool2d_kernel[grid](input, output, input_shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_clamp(input: torch.Tensor, clamp_min, clamp_max):
    input_shape = input.shape
    output = torch.empty(input_shape, device=input.device, dtype=input.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: ((input_shape[2] * input_shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    clamp_kernel[grid](input, output, clamp_min, clamp_max, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.scale_shape = scale_shape
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.gamma = nn.Parameter(torch.ones(*scale_shape))
        self.beta = nn.Parameter(torch.zeros(*scale_shape))

    def forward(self, x):
        x = triton_conv2d(x, self.weight, self.kernel_size, (1, 1), (0, 0))
        x = triton_group_norm(x, self.gamma, self.beta, self.num_groups, 1e-5)
        x = x * self.gamma
        x = triton_maxpool2d(x, self.maxpool_kernel_size, (2, 2), (0, 0))
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        return x