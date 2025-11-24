import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, depth, height, width)
    output_shape,  # (batch, out_channels, new_depth, new_height, new_width)
    kernel_size,  # (k_d, k_h, k_w)
    stride,  # (s_d, s_h, s_w)
    padding,  # (p_d, p_h, p_w)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Convert to 5D index
    out_idx_5d = tl.reshape(out_idx, (output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]))
    # Compute input index
    in_idx_5d = tl.reshape(out_idx_5d, (output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]))
    # Compute input indices based on transpose convolution
    in_idx_5d = tl.where(in_idx_5d[2] < padding[0], in_idx_5d[2] + padding[0], in_idx_5d[2] - (stride[0] - 1))
    in_idx_5d = tl.where(in_idx_5d[3] < padding[1], in_idx_5d[3] + padding[1], in_idx_5d[3] - (stride[1] - 1))
    in_idx_5d = tl.where(in_idx_5d[4] < padding[2], in_idx_5d[4] + padding[2], in_idx_5d[4] - (stride[2] - 1))
    # Compute input indices
    in_idx_5d = tl.where(in_idx_5d[2] < input_shape[2], in_idx_5d[2], in_idx_5d[2] - (stride[0] - 1))
    in_idx_5d = tl.where(in_idx_5d[3] < input_shape[3], in_idx_5d[3], in_idx_5d[3] - (stride[1] - 1))
    in_idx_5d = tl.where(in_idx_5d[4] < input_shape[4], in_idx_5d[4], in_idx_5d[4] - (stride[2] - 1))
    # Compute input indices
    in_idx_5d = tl.where(in_idx_5d[2] >= 0, in_idx_5d[2], in_idx_5d[2] + (stride[0] - 1))
    in_idx_5d = tl.where(in_idx_5d[3] >= 0, in_idx_5d[3], in_idx_5d[3] + (stride[1] - 1))
    in_idx_5d = tl.where(in_idx_5d[4] >= 0, in_idx_5d[4], in_idx_5d[4] + (stride[2] - 1))
    # Convert to 1D index
    in_idx = tl.reshape(in_idx_5d, (-1,))
    # Compute weight indices
    weight_idx = tl.arange(0, output_shape[1]) * (kernel_size[0] * kernel_size[1] * kernel_size[2]) + tl.arange(0, kernel_size[0] * kernel_size[1] * kernel_size[2])
    # Compute output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        input_val = tl.load(input_ptr + in_idx[i], mask=in_idx[i] < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx[i], mask=weight_idx[i] < weight_ptr.shape[0], other=0.0)
        output[i] += input_val * weight_val
    # Store output
    tl.store(output_ptr + out_idx, output, mask=out_idx < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4])


@triton.jit
def batch_norm_kernel(
    input_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    eps: tl.constexpr,
    momentum: tl.constexpr,
    gamma_ptr,  # Pointer to gamma tensor
    beta_ptr,  # Pointer to beta tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Compute the input index
    in_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Convert to 5D index
    in_idx_5d = tl.reshape(in_idx, (input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]))
    # Compute mean and variance
    mean = tl.load(mean_ptr, mask=mean_ptr < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
    var = tl.load(var_ptr, mask=var_ptr < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
    # Compute gamma and beta
    gamma = tl.load(gamma_ptr, mask=gamma_ptr < input_shape[1], other=1.0)
    beta = tl.load(beta_ptr, mask=beta_ptr < input_shape[1], other=0.0)
    # Compute output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        input_val = tl.load(input_ptr + in_idx[i], mask=in_idx[i] < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
        output[i] = (input_val - mean) * gamma / tl.sqrt(var + eps) + beta
    # Store output
    tl.store(output_ptr + in_idx, output, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4])


@triton.jit
def subtract_mean_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Compute the input index
    in_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Convert to 5D index
    in_idx_5d = tl.reshape(in_idx, (input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]))
    # Compute mean
    mean = tl.zeros((input_shape[1],), dtype=tl.float32)
    for i in range(input_shape[0]):
        for j in range(input_shape[2]):
            for k in range(input_shape[3]):
                for l in range(input_shape[4]):
                    val = tl.load(input_ptr + in_idx[i * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] + j * input_shape[1] * input_shape[3] * input_shape[4] + k * input_shape[1] * input_shape[4] + l * input_shape[1] + i], other=0.0)
                    mean += val
    mean /= (input_shape[0] * input_shape[2] * input_shape[3] * input_shape[4])
    # Compute output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        val = tl.load(input_ptr + in_idx[i], mask=in_idx[i] < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
        output[i] = val - mean
    # Store output
    tl.store(output_ptr + in_idx, output, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4])


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, kernel_size, stride, padding):
    """
    Custom Triton kernel for 3D convolution transpose.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Output shape calculation
    batch, in_channels, depth, height, width = x.shape
    out_channels = weight.shape[0]
    out_depth = (depth - 1) * stride[0] + kernel_size[0]
    out_height = (height - 1) * stride[1] + kernel_size[1]
    out_width = (width - 1) * stride[2] + kernel_size[2]
    output_shape = (batch, out_channels, out_depth, out_height, out_width)

    # Prepare output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_kernel[grid](x, weight, output, x.shape, output_shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_batch_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-5, momentum=0.1):
    """
    Custom Triton kernel for batch normalization.
    """
    assert x.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Input shape
    input_shape = x.shape

    # Prepare output tensor
    output = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batch_norm_kernel[grid](x, mean, var, output, input_shape, eps, momentum, gamma, beta, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_subtract_mean(x: torch.Tensor):
    """
    Custom Triton kernel for subtracting mean along spatial dimensions.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Input shape
    input_shape = x.shape

    # Prepare output tensor
    output = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    subtract_mean_kernel[grid](x, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1, 1))
        else:
            self.register_parameter('bias', None)

        # Initialize mean and variance for batch normalization
        self.mean = nn.Parameter(torch.zeros(out_channels, 1, 1, 1, 1))
        self.var = nn.Parameter(torch.ones(out_channels, 1, 1, 1, 1))

        # Initialize gamma and beta for batch normalization
        self.gamma = nn.Parameter(torch.ones(out_channels, 1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1, 1, 1))

    def forward(self, x):
        # Custom Triton convolution transpose
        x = triton_conv_transpose(x, self.weight, self.kernel_size, self.stride, self.padding)
        # Custom Triton batch normalization
        x = triton_batch_norm(x, self.mean, self.var, self.gamma, self.beta)
        # Custom Triton subtract mean
        x = triton_subtract_mean(x)
        return x