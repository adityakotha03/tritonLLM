import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # pointer to input
    w_ptr,  # pointer to weights
    out_ptr,  # pointer to output
    stride,  # stride of the convolution
    padding,  # padding of the convolution
    kernel_size,  # kernel size
    out_channels,  # number of output channels
    in_channels,  # number of input channels
    height,  # height of input
    width,  # width of input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of output
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // width
    out_w = pid % width

    # Compute the input position
    input_h_start = out_h * stride - padding
    input_w_start = out_w * stride - padding

    # Initialize the output value
    out_val = tl.zeros((out_channels,), dtype=tl.float32)

    # Iterate over the kernel
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute input position for this kernel element
            input_h = input_h_start + kh
            input_w = input_w_start + kw

            # Check if input is within bounds
            if input_h < 0 or input_h >= height or input_w < 0 or input_w >= width:
                continue

            # Compute the offset in input
            input_offset = (input_h * width + input_w) * in_channels
            # Load the input values
            input_vals = tl.load(x_ptr + input_offset, mask=tl.arange(0, in_channels), other=0.0)

            # Compute the weight offset
            weight_offset = (kh * kernel_size + kw) * in_channels * out_channels
            # Load the weights
            weights = tl.load(w_ptr + weight_offset, mask=tl.arange(0, in_channels * out_channels), other=0.0)

            # Perform the matrix multiplication
            out_val += tl.dot(input_vals, weights)

    # Store the result
    out_offset = (out_h * width + out_w) * out_channels
    tl.store(out_ptr + out_offset, out_val)


@triton.jit
def group_norm_kernel(
    x_ptr,  # pointer to input
    gamma_ptr,  # pointer to gamma
    beta_ptr,  # pointer to beta
    out_ptr,  # pointer to output
    num_groups,  # number of groups
    group_size,  # size of each group
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the group and position within group
    group_id = pid // group_size
    pos_in_group = pid % group_size

    # Compute the offset in input
    input_offset = group_id * group_size + pos_in_group
    # Load the input values
    x = tl.load(x_ptr + input_offset, mask=tl.arange(0, group_size), other=0.0)

    # Compute mean and variance
    mean = tl.mean(x)
    var = tl.var(x, mean=mean)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Scale and shift
    gamma = tl.load(gamma_ptr + group_id, mask=tl.arange(0, group_size), other=0.0)
    beta = tl.load(beta_ptr + group_id, mask=tl.arange(0, group_size), other=0.0)
    out = x_norm * gamma + beta

    # Store the result
    tl.store(out_ptr + input_offset, out, mask=tl.arange(0, group_size))


@triton.jit
def tanh_kernel(
    x_ptr,  # pointer to input
    out_ptr,  # pointer to output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the offset
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Compute tanh
    out = tl.tanh(x)

    # Store the result
    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def hard_swish_kernel(
    x_ptr,  # pointer to input
    out_ptr,  # pointer to output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the offset
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Compute hard_swish
    x = x * (tl.clamp(x + 3, 0, 6) / 6)
    out = x

    # Store the result
    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def residual_add_kernel(
    x1_ptr,  # pointer to first input
    x2_ptr,  # pointer to second input
    out_ptr,  # pointer to output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the offset
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    x1 = tl.load(x1_ptr + offset, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offset, mask=mask, other=0.0)

    # Perform addition
    out = x1 + x2

    # Store the result
    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr,  # pointer to input
    out_ptr,  # pointer to output
    n_elements,  # number of elements
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the offset
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Compute max along the specified dimension
    max_val = tl.max(x, axis=dim)
    # Compute logsumexp
    out = tl.log(tl.sum(tl.exp(x - max_val), axis=dim)) + max_val

    # Store the result
    tl.store(out_ptr + offset, out, mask=mask)


def triton_conv2d(x, w, stride, padding, kernel_size):
    out_channels = w.shape[0]
    in_channels = w.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((x.shape[0], out_channels, out_h, out_w), device=x.device, dtype=x.dtype)

    # Number of elements in the output
    n_elements = out.numel()
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, out, stride, padding, kernel_size, out_channels, in_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_group_norm(x, gamma, beta, num_groups, group_size, eps):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    group_norm_kernel[grid](x, gamma, beta, out, num_groups, group_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_tanh(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_hard_swish(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    hard_swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_residual_add(x1, x2):
    out = torch.empty_like(x1)
    n_elements = x1.numel()
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    residual_add_kernel[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_logsumexp(x, dim):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    logsumexp_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Convolution
        x_conv = triton_conv2d(x, self.weight, stride=1, padding=(self.kernel_size - 1) // 2, kernel_size=self.kernel_size)
        # Group Normalization
        x_norm = triton_group_norm(x_conv, self.gamma, self.beta, self.groups, self.out_channels // self.groups, self.eps)
        # Tanh
        x_tanh = triton_tanh(x_norm)
        # HardSwish
        x_hard_swish = triton_hard_swish(x_tanh)
        # Residual Addition
        x_res = triton_residual_add(x_conv, x_hard_swish)
        # LogSumExp
        x_logsumexp = triton_logsumexp(x_res, dim=1)
        return x_logsumexp