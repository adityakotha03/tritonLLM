import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,  # pointer to conv weights (out_channels, in_channels, kernel_size, kernel_size)
    bias_ptr,  # pointer to bias (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H, W)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Define grid and block dimensions
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width

    # Compute the output position
    output_h = h_idx
    output_w = w_idx
    batch = batch_idx

    # Compute the input and output indices
    input_h = output_h - (kernel_size // 2)
    input_w = output_w - (kernel_size // 2)

    # Clamp input indices to valid range
    input_h = tl.maximum(input_h, 0)
    input_h = tl.minimum(input_h, height - kernel_size)
    input_w = tl.maximum(input_w, 0)
    input_w = tl.minimum(input_w, width - kernel_size)

    # Load input patch
    input_offset = batch * in_channels * height * width + \
                   input_h * in_channels * width + \
                   input_w * in_channels
    input_vals = tl.load(input_ptr + input_offset, mask=tl.arange(0, in_channels) < in_channels, other=0.0)

    # Compute output
    output_val = 0.0
    for g in range(groups):
        # Group-wise convolution
        in_chan_start = g * (in_channels // groups)
        in_chan_end = (g + 1) * (in_channels // groups)
        if in_chan_end > in_channels:
            in_chan_end = in_channels

        # Load weights
        weight_offset = g * out_channels * in_channels * kernel_size * kernel_size + \
                        tl.arange(0, out_channels) * in_channels * kernel_size * kernel_size + \
                        in_chan_start * kernel_size * kernel_size + \
                        tl.arange(0, kernel_size) * kernel_size + tl.arange(0, kernel_size)
        weights = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, out_channels) < out_channels, other=0.0)

        # Convolve over kernel
        for i in range(kernel_size):
            for j in range(kernel_size):
                input_idx = input_h + i
                input_idx = tl.maximum(input_idx, 0)
                input_idx = tl.minimum(input_idx, height - 1)
                input_idx = tl.maximum(input_idx - (kernel_size // 2), 0)
                input_idx = tl.minimum(input_idx, height - kernel_size)
                input_idx = input_idx
                input_idx = input_idx * in_channels * width + input_w * in_channels + in_chan_start
                input_val = tl.load(input_ptr + input_idx, mask=tl.arange(0, in_channels) < in_channels, other=0.0)
                weight_idx = g * out_channels * in_channels * kernel_size * kernel_size + \
                             tl.arange(0, out_channels) * in_channels * kernel_size * kernel_size + \
                             in_chan_start * kernel_size * kernel_size + i * kernel_size + j
                weight_val = tl.load(weight_ptr + weight_idx, mask=tl.arange(0, out_channels) < out_channels, other=0.0)
                output_val += input_val * weight_val
    output_val = output_val + tl.load(bias_ptr + tl.arange(0, out_channels), mask=tl.arange(0, out_channels) < out_channels, other=0.0)

    # Store output
    output_offset = batch * out_channels * height * width + output_h * out_channels * width + output_w * out_channels
    tl.store(output_ptr + output_offset, output_val, mask=tl.arange(0, out_channels) < out_channels)


@triton.jit
def group_norm_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, H, W)
    gamma_ptr,  # pointer to gamma (channels)
    beta_ptr,  # pointer to beta (channels)
    output_ptr,  # pointer to output tensor (batch, channels, H, W)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width

    batch = batch_idx
    h = h_idx
    w = w_idx

    # Compute the channel index
    channel_idx = tl.arange(0, channels)
    group_size = channels // groups
    group_id = tl.arange(0, groups)
    group_idx = group_id * group_size + channel_idx

    # Load input
    input_offset = batch * channels * height * width + h * channels * width + w * channels
    x_vals = tl.load(x_ptr + input_offset, mask=tl.arange(0, channels) < channels, other=0.0)

    # Group-wise normalization
    x_group = x_vals[tl.arange(0, group_size)]
    mean = tl.sum(x_group, axis=0) / (group_size)
    var = tl.sum((x_group - mean) ** 2, axis=0) / (group_size)
    std = tl.sqrt(var + eps)
    inv_std = 1.0 / std
    x_norm = (x_group - mean) * inv_std

    # Apply gamma and beta
    gamma_vals = tl.load(gamma_ptr + group_idx, mask=tl.arange(0, channels) < channels, other=1.0)
    beta_vals = tl.load(beta_ptr + group_idx, mask=tl.arange(0, channels) < channels, other=0.0)
    output_vals = x_norm * gamma_vals + beta_vals

    # Store output
    output_offset = batch * channels * height * width + h * channels * width + w * channels
    tl.store(output_ptr + output_offset, output_vals, mask=tl.arange(0, channels) < channels)


@triton.jit
def tanh_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, H, W)
    output_ptr,  # pointer to output tensor (batch, channels, H, W)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width

    batch = batch_idx
    h = h_idx
    w = w_idx

    # Load input
    input_offset = batch * channels * height * width + h * channels * width + w * channels
    x_vals = tl.load(x_ptr + input_offset, mask=tl.arange(0, channels) < channels, other=0.0)

    # Apply Tanh
    tanh_vals = tl.tanh(x_vals)

    # Store output
    output_offset = batch * channels * height * width + h * channels * width + w * channels
    tl.store(output_ptr + output_offset, tanh_vals, mask=tl.arange(0, channels) < channels)


@triton.jit
def hard_swish_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, H, W)
    output_ptr,  # pointer to output tensor (batch, channels, H, W)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width

    batch = batch_idx
    h = h_idx
    w = w_idx

    # Load input
    input_offset = batch * channels * height * width + h * channels * width + w * channels
    x_vals = tl.load(x_ptr + input_offset, mask=tl.arange(0, channels) < channels, other=0.0)

    # Apply HardSwish: x * (x + 3) / 6
    x_pos = tl.maximum(x_vals, 0.0)
    x_neg = tl.minimum(x_vals, 0.0)
    hard_swish_vals = (x_pos * (x_pos + 3.0) + x_neg * (x_neg + 3.0)) / 6.0

    # Store output
    output_offset = batch * channels * height * width + h * channels * width + w * channels
    tl.store(output_ptr + output_offset, hard_swish_vals, mask=tl.arange(0, channels) < channels)


@triton.jit
def residual_add_kernel(
    x1_ptr,  # pointer to first input (conv output)
    x2_ptr,  # pointer to second input (hard_swish output)
    output_ptr,  # pointer to output (residual addition)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width

    batch = batch_idx
    h = h_idx
    w = w_idx

    # Load both inputs
    x1_offset = batch * channels * height * width + h * channels * width + w * channels
    x1_val = tl.load(x1_ptr + x1_offset, mask=tl.arange(0, channels) < channels, other=0.0)
    x2_offset = batch * channels * height * width + h * channels * width + w * channels
    x2_val = tl.load(x2_ptr + x2_offset, mask=tl.arange(0, channels) < channels, other=0.0)

    # Add
    out_val = x1_val + x2_val

    # Store
    output_offset = batch * channels * height * width + h * channels * width + w * channels
    tl.store(output_ptr + output_offset, out_val, mask=tl.arange(0, channels) < channels)


@triton.jit
def logsumexp_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, H, W)
    output_ptr,  # pointer to output tensor (batch, 1, H, W)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width

    batch = batch_idx
    h = h_idx
    w = w_idx

    # Load input
    input_offset = batch * channels * height * width + h * channels * width + w * channels
    x_vals = tl.load(x_ptr + input_offset, mask=tl.arange(0, channels) < channels, other=0.0)

    # Compute logsumexp over channels
    max_val = tl.max(x_vals, axis=0)
    x_shifted = x_vals - max_val
    exp_vals = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    logsumexp_val = max_val + tl.log(sum_exp)

    # Store result
    output_offset = batch * 1 * height * width + h * width + w
    tl.store(output_ptr + output_offset, logsumexp_val, mask=tl.arange(0, 1) < 1)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    groups = weight.shape[1] // in_channels  # assuming correct group count

    # Use FP16 for tensor core acceleration
    x = x.to(torch.float16)
    weight = weight.to(torch.float16)
    bias = bias.to(torch.float16)

    # Define block size
    BLOCK_SIZE = 128

    # Grid: number of blocks needed
    grid = lambda meta: (
        (batch_size * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr(),
        output_ptr=torch.empty_like(x).data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty_like(x)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    batch_size, channels, height, width = x.shape

    # Use FP16 for tensor core
    x = x.to(torch.float16)
    gamma = gamma.to(torch.float16)
    beta = beta.to(torch.float16)

    BLOCK_SIZE = 128
    grid = lambda meta: (
        (batch_size * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    group_norm_kernel[grid](
        x_ptr=x.data_ptr(),
        gamma_ptr=gamma.data_ptr(),
        beta_ptr=beta.data_ptr(),
        output_ptr=torch.empty_like(x).data_ptr(),
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        groups=groups,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty_like(x)


def triton_tanh(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    x = x.to(torch.float16)

    BLOCK_SIZE = 128
    grid = lambda meta: (
        (x.shape[0] * x.shape[2] * x.shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    tanh_kernel[grid](
        x_ptr=x.data_ptr(),
        output_ptr=torch.empty_like(x).data_ptr(),
        batch_size=x.shape[0],
        channels=x.shape[1],
        height=x.shape[2],
        width=x.shape[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty_like(x)


def triton_hard_swish(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    x = x.to(torch.float16)

    BLOCK_SIZE = 128
    grid = lambda meta: (
        (x.shape[0] * x.shape[2] * x.shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    hard_swish_kernel[grid](
        x_ptr=x.data_ptr(),
        output_ptr=torch.empty_like(x).data_ptr(),
        batch_size=x.shape[0],
        channels=x.shape[1],
        height=x.shape[2],
        width=x.shape[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty_like(x)


def triton_residual_add(x1: torch.Tensor, x2: torch.Tensor):
    assert x1.is_cuda and x2.is_cuda, "Inputs must be on CUDA."
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    x1 = x1.to(torch.float16)
    x2 = x2.to(torch.float16)

    BLOCK_SIZE = 128
    grid = lambda meta: (
        (x1.shape[0] * x1.shape[2] * x1.shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    residual_add_kernel[grid](
        x1_ptr=x1.data_ptr(),
        x2_ptr=x2.data_ptr(),
        output_ptr=torch.empty_like(x1).data_ptr(),
        batch_size=x1.shape[0],
        channels=x1.shape[1],
        height=x1.shape[2],
        width=x1.shape[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty_like(x1)


def triton_logsumexp(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    x = x.to(torch.float16)

    batch_size, channels, height, width = x.shape

    BLOCK_SIZE = 128
    grid = lambda meta: (
        (batch_size * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    logsumexp_kernel[grid](
        x_ptr=x.data_ptr(),
        output_ptr=torch.empty(batch_size, 1, height, width, device=x.device).data_ptr(),
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty(batch_size, 1, height, width, device=x.device)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # Initialize weights and biases
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        self.group_norm_gamma = nn.Parameter(torch.ones(out_channels))
        self.group_norm_beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Convolution
        x_conv = triton_conv2d(x, self.conv_weight, self.conv_bias)
        # Group Normalization
        x_norm = triton_group_norm(x_conv, self.group_norm_gamma, self.group_norm_beta)
        # Tanh
        x_tanh = triton_tanh(x_norm)
        # HardSwish
        x_hard_swish = triton_hard_swish(x_tanh)
        # Residual Addition
        x_res = triton_residual_add(x_conv, x_hard_swish)
        # LogSumExp
        x_logsumexp = triton_logsumexp(x_res)
        return x_logsumexp