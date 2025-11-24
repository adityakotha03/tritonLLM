import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    x_ptr,           # Input tensor pointer
    scale_ptr,       # Scale parameter pointer (for batch norm)
    bias_ptr,        # Bias parameter pointer
    mean_ptr,        # Mean parameter pointer
    var_ptr,         # Variance parameter pointer
    eps,             # Small value for numerical stability
    n_elements,      # Total number of elements
    BLOCK_SIZE: tl.constexpr,
    num_channels: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load scale, bias, mean, variance
    scale = tl.load(scale_ptr + tl.arange(0, num_channels), mask=tl.arange(0, num_channels) < num_channels, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, num_channels), mask=tl.arange(0, num_channels) < num_channels, other=0.0)
    mean = tl.load(mean_ptr + tl.arange(0, num_channels), mask=tl.arange(0, num_channels) < num_channels, other=0.0)
    var = tl.load(var_ptr + tl.arange(0, num_channels), mask=tl.arange(0, num_channels) < num_channels, other=1.0)

    # Compute batch norm: (x - mean) / sqrt(var + eps) * scale + bias
    # Vectorized per channel
    x_norm = (x - mean) / tl.sqrt(var + eps)
    output = x_norm * scale + bias

    # Store result
    tl.store(x_ptr + offsets, output, mask=mask)


@triton.jit
def relu_kernel(
    x_ptr,            # Input pointer
    out_ptr,          # Output pointer
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def conv2d_kernel(
    input_ptr,        # Input tensor pointer (batch, channels, height, width)
    weight_ptr,       # Weight tensor pointer (out_channels, in_channels, 3, 3)
    bias_ptr,         # Bias pointer (out_channels)
    output_ptr,       # Output tensor pointer (batch, out_channels, height, width)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Define grid dimensions
    pid = tl.program_id(0)
    block_h = pid // (height // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)

    # Compute output position
    h_start = block_h * BLOCK_SIZE
    w_start = block_w * BLOCK_SIZE
    h_end = min(h_start + BLOCK_SIZE, height)
    w_end = min(w_start + BLOCK_SIZE, width)

    # Compute output indices
    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)
    h = h_idx + h_start
    w = w_idx + w_start

    # Compute input indices with padding
    h_input = h - padding
    w_input = w - padding
    h_input = h_input + tl.arange(0, BLOCK_SIZE)
    w_input = w_input + tl.arange(0, BLOCK_SIZE)

    # Compute kernel indices
    k_h = tl.arange(0, kernel_size)
    k_w = tl.arange(0, kernel_size)

    # Load input and weights
    input_batch = tl.arange(0, batch_size)
    input_channel = tl.arange(0, in_channels)
    output_channel = tl.arange(0, out_channels)

    # Initialize output
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Compute convolution via tiling
    for i in tl.arange(0, in_channels):
        for j in tl.arange(0, out_channels):
            # Load weight
            w = tl.load(weight_ptr + j * (in_channels * kernel_size * kernel_size) + i * (kernel_size * kernel_size) + k_h[:, None] * kernel_size + k_w[None, :], mask=(k_h < kernel_size) & (k_w < kernel_size), other=0.0)

            # Compute input
            input_vals = tl.load(input_ptr + (input_batch[:, None] * in_channels * height * width) + (i * height * width) + (h_input[:, :, None] * width + w_input[:, :, None]), mask=(h_input < height) & (w_input < width), other=0.0)

            # Perform convolution
            conv_output = tl.sum(input_vals * w, axis=(1, 2))
            output += conv_output

    # Add bias
    bias = tl.load(bias_ptr + j, mask=(j < out_channels), other=0.0)
    output += bias

    # Store output
    tl.store(output_ptr + (pid * BLOCK_SIZE * BLOCK_SIZE), output, mask=(h < height) & (w < width))


@triton.jit
def avg_pool2d_kernel(
    x_ptr,            # Input pointer
    output_ptr,       # Output pointer
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_h = pid // (height // BLOCK_SIZE)
    block_w = pid % (width // BLOCK_SIZE)

    h_start = block_h * BLOCK_SIZE
    w_start = block_w * BLOCK_SIZE
    h_end = min(h_start + BLOCK_SIZE, height)
    w_end = min(w_start + BLOCK_SIZE, width)

    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)
    h = h_idx + h_start
    w = w_idx + w_start

    # Load input
    input_vals = tl.load(x_ptr + (tl.arange(0, batch_size)[:, None] * channels * height * width) + (tl.arange(0, channels)[:, None] * height * width) + (h[:, :, None] * width + w[:, :, None]), mask=(h < height) & (w < width), other=0.0)

    # Compute average over kernel region
    count = tl.sum(tl.ones_like(input_vals), axis=(1, 2))
    avg = tl.sum(input_vals, axis=(1, 2)) / count

    # Store output
    tl.store(output_ptr + (pid * BLOCK_SIZE * BLOCK_SIZE), avg, mask=(h < height) & (w < width))


def triton_batch_norm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and scale.is_cuda and bias.is_cuda and mean.is_cuda and var.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    scale = scale.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    n_elements = x.numel()
    num_channels = x.shape[1]
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    batch_norm_kernel[grid](x, scale, bias, mean, var, eps, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_channels=num_channels)
    return x


def triton_relu(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


def triton_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    kernel_size: int = 3,
) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    BLOCK_SIZE = 128

    # Use FP16 for better performance on Tensor Cores
    input = input.half()
    weight = weight.half()
    bias = bias.half()

    grid = lambda meta: ((height // BLOCK_SIZE + 1) * (width // BLOCK_SIZE + 1),)

    output = torch.empty_like(input, dtype=torch.float16)

    conv2d_kernel[grid](
        input, weight, bias, output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.to(torch.float32)


def triton_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int = 2,
    stride: int = 2,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    BLOCK_SIZE = 128

    grid = lambda meta: ((height // BLOCK_SIZE + 1) * (width // BLOCK_SIZE + 1),)

    output = torch.empty(batch_size, channels, height // stride, width // stride, dtype=torch.float32)

    avg_pool2d_kernel[grid](
        x, output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 48, 32]  # Corresponding layers in DenseNet201

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x