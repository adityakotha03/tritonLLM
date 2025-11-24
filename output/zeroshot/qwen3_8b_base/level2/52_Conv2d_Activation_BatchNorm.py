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
    weight_shape,  # (O, I, K, K)
    output_shape,  # (N, O, H, W)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block offset
    block_id = pid
    # Compute the output position
    n, o, h, w = block_id // (output_shape[1] * output_shape[2] * output_shape[3]), \
                 block_id % (output_shape[1] * output_shape[2] * output_shape[3]) // (output_shape[2] * output_shape[3]), \
                 block_id % (output_shape[2] * output_shape[3]) // output_shape[3], \
                 block_id % output_shape[3]

    # Compute the input position
    i_h_start = h * stride[0] - padding[0]
    i_w_start = w * stride[1] - padding[1]
    i_h_end = i_h_start + weight_shape[2]
    i_w_end = i_w_start + weight_shape[3]

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the input channels
    for c in range(weight_shape[1]):
        # Load the weight
        weight = tl.load(weight_ptr + c * weight_shape[2] * weight_shape[3] + tl.arange(0, weight_shape[2]) * weight_shape[3] + tl.arange(0, weight_shape[3]), mask=(tl.arange(0, weight_shape[2]) < weight_shape[2]) & (tl.arange(0, weight_shape[3]) < weight_shape[3]))

        # Iterate over the kernel
        for k_h in range(weight_shape[2]):
            for k_w in range(weight_shape[3]):
                # Compute the input position
                i_h = i_h_start + k_h
                i_w = i_w_start + k_w

                # Check if input position is valid
                if i_h < 0 or i_h >= input_shape[2] or i_w < 0 or i_w >= input_shape[3]:
                    continue

                # Load the input value
                input_val = tl.load(input_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + i_h * input_shape[3] + i_w, mask=(i_h >= 0) & (i_h < input_shape[2]) & (i_w >= 0) & (i_w < input_shape[3]), other=0.0)

                # Multiply and accumulate
                acc += input_val * weight[k_h * weight_shape[3] + k_w]

    # Store the result
    output = tl.load(output_ptr + n * output_shape[1] * output_shape[2] * output_shape[3] + o * output_shape[2] * output_shape[3] + h * output_shape[3] + w, mask=(h >= 0) & (h < output_shape[2]) & (w >= 0) & (w < output_shape[3]), other=0.0)
    output += acc
    tl.store(output_ptr + n * output_shape[1] * output_shape[2] * output_shape[3] + o * output_shape[2] * output_shape[3] + h * output_shape[3] + w, output)


@triton.jit
def bn_kernel(
    input_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean
    var_ptr,  # Pointer to variance
    gamma_ptr,  # Pointer to gamma
    beta_ptr,  # Pointer to beta
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block offset
    block_id = pid
    # Compute the output position
    n, c, h, w = block_id // (input_shape[1] * input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[1] * input_shape[2] * input_shape[3]) // (input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[2] * input_shape[3]) // input_shape[3], \
                 block_id % input_shape[3]

    # Load input value
    input_val = tl.load(input_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)

    # Compute mean and variance (not implemented for this kernel)
    # For simplicity, we assume mean and variance are precomputed
    mean = tl.load(mean_ptr + c, mask=(c >= 0) & (c < input_shape[1]), other=0.0)
    var = tl.load(var_ptr + c, mask=(c >= 0) & (c < input_shape[1]), other=0.0)

    # Normalize
    normalized = (input_val - mean) / tl.sqrt(var + eps)

    # Apply gamma and beta
    gamma = tl.load(gamma_ptr + c, mask=(c >= 0) & (c < input_shape[1]), other=0.0)
    beta = tl.load(beta_ptr + c, mask=(c >= 0) & (c < input_shape[1]), other=0.0)

    output_val = normalized * gamma + beta

    # Store the result
    tl.store(output_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, output_val)


@triton.jit
def softplus_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block offset
    block_id = pid
    # Compute the output position
    n, c, h, w = block_id // (input_shape[1] * input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[1] * input_shape[2] * input_shape[3]) // (input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[2] * input_shape[3]) // input_shape[3], \
                 block_id % input_shape[3]

    # Load input value
    input_val = tl.load(input_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)

    # Compute softplus
    output_val = tl.maximum(input_val, tl.log(1.0 + tl.exp(-input_val)))

    # Store the result
    tl.store(output_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, output_val)


@triton.jit
def tanh_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block offset
    block_id = pid
    # Compute the output position
    n, c, h, w = block_id // (input_shape[1] * input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[1] * input_shape[2] * input_shape[3]) // (input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[2] * input_shape[3]) // input_shape[3], \
                 block_id % input_shape[3]

    # Load input value
    input_val = tl.load(input_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)

    # Compute tanh
    output_val = (tl.exp(2.0 * input_val) - 1.0) / (tl.exp(2.0 * input_val) + 1.0)

    # Store the result
    tl.store(output_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, output_val)


@triton.jit
def multiply_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread ID
    pid = tl.program_id(0)
    # Get the block offset
    block_id = pid
    # Compute the output position
    n, c, h, w = block_id // (input_shape[1] * input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[1] * input_shape[2] * input_shape[3]) // (input_shape[2] * input_shape[3]), \
                 block_id % (input_shape[2] * input_shape[3]) // input_shape[3], \
                 block_id % input_shape[3]

    # Load x and y values
    x = tl.load(x_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)
    y = tl.load(y_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, mask=(h >= 0) & (h < input_shape[2]) & (w >= 0) & (w < input_shape[3]), other=0.0)

    # Multiply
    out = x * y

    # Store the result
    tl.store(out_ptr + n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w, out)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, stride: tuple, padding: tuple, BLOCK_SIZE=128):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Output shape
    input_shape = x.shape
    weight_shape = weight.shape
    output_shape = (input_shape[0], weight_shape[0], (input_shape[2] + 2 * padding[0] - weight_shape[2]) // stride[0] + 1, (input_shape[3] + 2 * padding[1] - weight_shape[3]) // stride[1] + 1)

    # Prepare output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # Determine the number of blocks needed
    num_blocks = (input_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    conv2d_kernel[triton.make_kernel(num_blocks)](x, weight, output, input_shape, weight_shape, output_shape, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_bn(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-5, BLOCK_SIZE=128):
    assert x.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Output shape
    input_shape = x.shape

    # Prepare output tensor
    output = torch.empty_like(x)

    # Determine the number of blocks needed
    num_blocks = (input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    bn_kernel[triton.make_kernel(num_blocks)](x, mean, var, gamma, beta, output, input_shape, eps=eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_softplus(x: torch.Tensor, BLOCK_SIZE=128):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Output shape
    input_shape = x.shape

    # Prepare output tensor
    output = torch.empty_like(x)

    # Determine the number of blocks needed
    num_blocks = (input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    softplus_kernel[triton.make_kernel(num_blocks)](x, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_tanh(x: torch.Tensor, BLOCK_SIZE=128):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Output shape
    input_shape = x.shape

    # Prepare output tensor
    output = torch.empty_like(x)

    # Determine the number of blocks needed
    num_blocks = (input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    tanh_kernel[triton.make_kernel(num_blocks)](x, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_multiply(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE=128):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Output shape
    input_shape = x.shape

    # Prepare output tensor
    output = torch.empty_like(x)

    # Determine the number of blocks needed
    num_blocks = (input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    multiply_kernel[triton.make_kernel(num_blocks)](x, y, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.momentum = momentum
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda())
        self.gamma = torch.nn.Parameter(torch.ones(out_channels).cuda())
        self.beta = torch.nn.Parameter(torch.zeros(out_channels).cuda())
        self.mean = torch.zeros(out_channels).cuda()
        self.var = torch.ones(out_channels).cuda()

    def forward(self, x):
        # Custom conv2d
        x = triton_conv2d(x, self.weight, stride=(1, 1), padding=(1, 1))
        # Custom softplus
        x = triton_softplus(x)
        # Custom tanh
        x = triton_tanh(x)
        # Custom multiply
        x = triton_multiply(x, x)
        # Custom batch normalization
        x = triton_bn(x, self.mean, self.var, self.gamma, self.beta, self.eps)
        return x