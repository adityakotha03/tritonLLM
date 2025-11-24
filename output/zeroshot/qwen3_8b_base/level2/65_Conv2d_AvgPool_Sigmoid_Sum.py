import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # [N, C, H, W]
    kernel_size,  # Kernel size (same for height and width)
    stride,  # Stride for convolution
    padding,  # Padding for convolution
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 4D index (n, c, h, w)
    pid = tl.program_id(0)
    n = pid // (input_shape[1] * input_shape[2] * input_shape[3])
    c = (pid // (input_shape[2] * input_shape[3])) % input_shape[1]
    h = (pid // input_shape[3]) % input_shape[2]
    w = pid % input_shape[3]

    # Compute the output position
    h_out = (h + padding) // stride
    w_out = (w + padding) // stride

    # Compute the output index
    out_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h_out * input_shape[3] + w_out

    # Compute the input indices for the current position
    h_start = h - padding
    w_start = w - padding
    h_end = h_start + kernel_size
    w_end = w_start + kernel_size

    # Compute the output value
    output = 0.0
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            c_in = c
            h_in = h_start + kh
            w_in = w_start + kw
            if h_in < 0 or h_in >= input_shape[2] or w_in < 0 or w_in >= input_shape[3]:
                continue
            input_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c_in * input_shape[2] * input_shape[3] + h_in * input_shape[3] + w_in
            weight_idx = c_in * out_channels * kernel_size * kernel_size + kh * kernel_size + kw
            input_val = tl.load(input_ptr + input_idx, mask=(h_in >= 0) & (h_in < input_shape[2]) & (w_in >= 0) & (w_in < input_shape[3]), other=0.0)
            weight_val = tl.load(weight_ptr + weight_idx, mask=(kh >= 0) & (kh < kernel_size) & (kw >= 0) & (kw < kernel_size), other=0.0)
            output += input_val * weight_val

    # Store the output
    tl.store(output_ptr + out_idx, output)


@triton.jit
def avg_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # [N, C, H, W]
    pool_kernel_size,  # Pool kernel size (same for height and width)
    stride,  # Stride for pooling
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 4D index (n, c, h, w)
    pid = tl.program_id(0)
    n = pid // (input_shape[1] * input_shape[2] * input_shape[3])
    c = (pid // (input_shape[2] * input_shape[3])) % input_shape[1]
    h = (pid // input_shape[3]) % input_shape[2]
    w = pid % input_shape[3]

    # Compute the output position
    h_out = h // stride
    w_out = w // stride

    # Compute the output index
    out_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h_out * input_shape[3] + w_out

    # Compute the input indices for the current position
    h_start = h - (pool_kernel_size // 2)
    w_start = w - (pool_kernel_size // 2)
    h_end = h_start + pool_kernel_size
    w_end = w_start + pool_kernel_size

    # Compute the average value
    total = 0.0
    count = 0
    for kh in range(pool_kernel_size):
        for kw in range(pool_kernel_size):
            h_in = h_start + kh
            w_in = w_start + kw
            if h_in < 0 or h_in >= input_shape[2] or w_in < 0 or w_in >= input_shape[3]:
                continue
            input_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h_in * input_shape[3] + w_in
            input_val = tl.load(input_ptr + input_idx, mask=(h_in >= 0) & (h_in < input_shape[2]) & (w_in >= 0) & (w_in < input_shape[3]), other=0.0)
            total += input_val
            count += 1

    # Store the average value
    tl.store(output_ptr + out_idx, total / count)


@triton.jit
def sigmoid_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # [N, C, H, W]
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 4D index (n, c, h, w)
    pid = tl.program_id(0)
    n = pid // (input_shape[1] * input_shape[2] * input_shape[3])
    c = (pid // (input_shape[2] * input_shape[3])) % input_shape[1]
    h = (pid // input_shape[3]) % input_shape[2]
    w = pid % input_shape[3]

    # Compute the input index
    input_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w

    # Compute the sigmoid value
    input_val = tl.load(input_ptr + input_idx, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)
    output_val = 1.0 / (1.0 + tl.exp(-input_val))

    # Store the output
    tl.store(output_ptr + input_idx, output_val)


@triton.jit
def sum_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # [N, C, H, W]
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 4D index (n, c, h, w)
    pid = tl.program_id(0)
    n = pid // (input_shape[1] * input_shape[2] * input_shape[3])
    c = (pid // (input_shape[2] * input_shape[3])) % input_shape[1]
    h = (pid // input_shape[3]) % input_shape[2]
    w = pid % input_shape[3]

    # Compute the input index
    input_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w

    # Compute the sum value
    input_val = tl.load(input_ptr + input_idx, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)
    output_val = input_val

    # Store the output
    tl.store(output_ptr + input_idx, output_val)


def triton_conv2d(input, weight, kernel_size, stride, padding):
    """
    Perform a 2D convolution using a custom Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty(input.shape[0], weight.shape[0], input.shape[2] // stride, input.shape[3] // stride, device=input.device)

    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    kernel_size = kernel_size
    stride = stride
    padding = padding

    # Determine the number of blocks needed
    n_elements = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_avg_pool2d(input, pool_kernel_size, stride):
    """
    Perform 2D average pooling using a custom Triton kernel.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty(input.shape[0], input.shape[1], input.shape[2] // stride, input.shape[3] // stride, device=input.device)

    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    pool_kernel_size = pool_kernel_size
    stride = stride

    # Determine the number of blocks needed
    n_elements = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    avg_pool2d_kernel[grid](input, output, input_shape, pool_kernel_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_sigmoid(input):
    """
    Apply sigmoid function using a custom Triton kernel.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])

    # Determine the number of blocks needed
    n_elements = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sigmoid_kernel[grid](input, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_sum(input):
    """
    Apply sum over all spatial dimensions using a custom Triton kernel.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty(input.shape[0], device=input.device)

    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])

    # Determine the number of blocks needed
    n_elements = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sum_kernel[grid](input, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Custom convolution
        x = triton_conv2d(x, torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).cuda(), self.kernel_size, 1, 1)
        # Custom average pooling
        x = triton_avg_pool2d(x, self.pool_kernel_size, 2)
        # Custom sigmoid
        x = triton_sigmoid(x)
        # Custom sum
        x = triton_sum(x)
        return x