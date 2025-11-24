import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, depth, height, width)
    weight_shape,  # (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
    output_shape,  # (batch, out_channels, depth, height, width)
    stride_depth, stride_height, stride_width,
    padding_depth, padding_height, padding_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the output position
    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_idx = tl.reshape(out_idx, (BLOCK_SIZE,))
    out_batch, out_channel = out_idx // (output_shape[2] * output_shape[3] * output_shape[4]), out_idx % (output_shape[2] * output_shape[3] * output_shape[4])
    out_depth, out_height, out_width = out_channel // (output_shape[3] * output_shape[4]), out_channel % (output_shape[3] * output_shape[4]), out_channel % output_shape[4]

    # Compute input indices
    in_depth_start = out_depth * stride_depth - padding_depth
    in_height_start = out_height * stride_height - padding_height
    in_width_start = out_width * stride_width - padding_width

    # Compute the input indices for the kernel
    in_depth = in_depth_start + tl.arange(0, weight_shape[2])
    in_height = in_height_start + tl.arange(0, weight_shape[3])
    in_width = in_width_start + tl.arange(0, weight_shape[4])

    # Compute the input positions
    in_idx = tl.arange(0, input_shape[1])  # in_channels
    in_idx = tl.reshape(in_idx, (input_shape[1], 1, 1, 1, 1))
    in_idx = tl.broadcast_to(in_idx, (input_shape[1], weight_shape[2], weight_shape[3], weight_shape[4], 1))
    in_idx = tl.reshape(in_idx, (input_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4], 1))
    in_idx = tl.reshape(in_idx, (input_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4], 1))

    # Compute the output positions
    out_idx = tl.arange(0, output_shape[1])  # out_channels
    out_idx = tl.reshape(out_idx, (output_shape[1], 1, 1, 1, 1))
    out_idx = tl.broadcast_to(out_idx, (output_shape[1], weight_shape[2], weight_shape[3], weight_shape[4], 1))
    out_idx = tl.reshape(out_idx, (output_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4], 1))

    # Compute the input and weight indices
    in_idx = in_idx + tl.reshape(in_depth, (1, weight_shape[2], 1, 1, 1)) * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]
    in_idx = in_idx + tl.reshape(in_height, (1, 1, weight_shape[3], 1, 1)) * input_shape[1] * input_shape[2] * input_shape[4]
    in_idx = in_idx + tl.reshape(in_width, (1, 1, 1, weight_shape[4], 1)) * input_shape[1] * input_shape[2]
    in_idx = in_idx + tl.reshape(out_batch, (1, 1, 1, 1, 1)) * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]
    in_idx = in_idx + tl.reshape(out_idx, (1, 1, 1, 1, 1)) * input_shape[2] * input_shape[3] * input_shape[4]

    # Load input and weight
    input_values = tl.load(input_ptr + in_idx, mask=tl.arange(0, input_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4]) < input_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4], other=0.0)
    weight_values = tl.load(weight_ptr + out_idx, mask=tl.arange(0, output_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4]) < output_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4], other=0.0)

    # Compute the convolution
    output = tl.sum(input_values * weight_values, axis=0)

    # Store the result
    tl.store(output_ptr + out_idx, output, mask=out_idx < output_shape[1])


@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the input position
    in_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_idx = tl.reshape(in_idx, (BLOCK_SIZE,))
    in_batch, in_channel = in_idx // (input_shape[2] * input_shape[3] * input_shape[4]), in_idx % (input_shape[2] * input_shape[3] * input_shape[4])
    in_depth, in_height, in_width = in_channel // (input_shape[3] * input_shape[4]), in_channel % (input_shape[3] * input_shape[4]), in_channel % input_shape[4]

    # Load input values
    input_values = tl.load(input_ptr + in_idx, mask=tl.arange(0, input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]) < input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)

    # Compute the max value
    max_val = tl.max(input_values)

    # Subtract the max value
    input_values = input_values - max_val

    # Compute the exponential
    exp_values = tl.exp(input_values)

    # Compute the sum
    sum_val = tl.sum(exp_values)

    # Compute the softmax
    output_values = exp_values / sum_val

    # Store the result
    tl.store(output_ptr + in_idx, output_values, mask=in_idx < input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4])


@triton.jit
def max_pool3d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    pool_shape,  # (kernel_depth, kernel_height, kernel_width)
    stride_depth, stride_height, stride_width,
    padding_depth, padding_height, padding_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    # Compute the output position
    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_idx = tl.reshape(out_idx, (BLOCK_SIZE,))
    out_batch, out_channel = out_idx // (input_shape[2] * input_shape[3] * input_shape[4]), out_idx % (input_shape[2] * input_shape[3] * input_shape[4])
    out_depth, out_height, out_width = out_channel // (input_shape[3] * input_shape[4]), out_channel % (input_shape[3] * input_shape[4]), out_channel % input_shape[4]

    # Compute input indices
    in_depth_start = out_depth * stride_depth - padding_depth
    in_height_start = out_height * stride_height - padding_height
    in_width_start = out_width * stride_width - padding_width

    # Compute the input indices for the pool
    in_depth = in_depth_start + tl.arange(0, pool_shape[0])
    in_height = in_height_start + tl.arange(0, pool_shape[1])
    in_width = in_width_start + tl.arange(0, pool_shape[2])

    # Compute the input positions
    in_idx = tl.arange(0, input_shape[1])  # in_channels
    in_idx = tl.reshape(in_idx, (input_shape[1], 1, 1, 1, 1))
    in_idx = tl.broadcast_to(in_idx, (input_shape[1], pool_shape[0], pool_shape[1], pool_shape[2], 1))
    in_idx = tl.reshape(in_idx, (input_shape[1] * pool_shape[0] * pool_shape[1] * pool_shape[2], 1))
    in_idx = tl.reshape(in_idx, (input_shape[1] * pool_shape[0] * pool_shape[1] * pool_shape[2], 1))

    # Compute the input positions
    in_idx = in_idx + tl.reshape(in_depth, (1, pool_shape[0], 1, 1, 1)) * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]
    in_idx = in_idx + tl.reshape(in_height, (1, 1, pool_shape[1], 1, 1)) * input_shape[1] * input_shape[2] * input_shape[4]
    in_idx = in_idx + tl.reshape(in_width, (1, 1, 1, pool_shape[2], 1)) * input_shape[1] * input_shape[2]
    in_idx = in_idx + tl.reshape(out_batch, (1, 1, 1, 1, 1)) * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]
    in_idx = in_idx + tl.reshape(out_channel, (1, 1, 1, 1, 1)) * input_shape[2] * input_shape[3] * input_shape[4]

    # Load input values
    input_values = tl.load(input_ptr + in_idx, mask=tl.arange(0, input_shape[1] * pool_shape[0] * pool_shape[1] * pool_shape[2]) < input_shape[1] * pool_shape[0] * pool_shape[1] * pool_shape[2], other=-float('inf'))

    # Compute the max value
    max_val = tl.max(input_values)

    # Store the result
    tl.store(output_ptr + out_idx, max_val, mask=out_idx < input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4])


def triton_conv3d(input, weight, stride_depth, stride_height, stride_width, padding_depth, padding_height, padding_width):
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output_shape = (input.shape[0], weight.shape[0], (input.shape[2] + 2 * padding_depth - 1) // stride_depth,
                    (input.shape[3] + 2 * padding_height - 1) // stride_height,
                    (input.shape[4] + 2 * padding_width - 1) // stride_width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3d_kernel[grid](input, weight, output, input.shape, weight.shape, output.shape,
                        stride_depth, stride_height, stride_width, padding_depth, padding_height, padding_width, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_softmax(input):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softmax_kernel[grid](input, output, input.shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_max_pool3d(input, pool_shape, stride_depth, stride_height, stride_width, padding_depth, padding_height, padding_width):
    # Ensure input is on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()

    # Prepare output tensor
    output_shape = (input.shape[0], input.shape[1], (input.shape[2] + 2 * padding_depth - 1) // stride_depth,
                    (input.shape[3] + 2 * padding_height - 1) // stride_height,
                    (input.shape[4] + 2 * padding_width - 1) // stride_width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    max_pool3d_kernel[grid](input, output, input.shape, pool_shape, stride_depth, stride_height, stride_width,
                            padding_depth, padding_height, padding_width, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Perform 3D convolution
        x = triton_conv3d(x, self.weight, self.stride_depth, self.stride_height, self.stride_width,
                          self.padding_depth, self.padding_height, self.padding_width)
        # Apply softmax
        x = triton_softmax(x)
        # Perform first max pooling
        x = triton_max_pool3d(x, (self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size),
                              self.stride_depth, self.stride_height, self.stride_width,
                              self.padding_depth, self.padding_height, self.padding_width)
        # Perform second max pooling
        x = triton_max_pool3d(x, (self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size),
                              self.stride_depth, self.stride_height, self.stride_width,
                              self.padding_depth, self.padding_height, self.padding_width)
        return x