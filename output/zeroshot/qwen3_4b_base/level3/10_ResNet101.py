import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (B, C_in, H, W)
    weight_ptr,  # pointer to convolutional weight (C_out, C_in, 3, 3)
    bias_ptr,  # pointer to bias (C_out)
    output_ptr,  # pointer to output (B, C_out, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_h: tl.constexpr,
    input_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block and thread indices
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Compute the output position
    out_h_start = out_h * BLOCK_SIZE
    out_w_start = out_w * BLOCK_SIZE
    out_h_end = out_h_start + BLOCK_SIZE
    out_w_end = out_w_start + BLOCK_SIZE

    # Clip to valid output bounds
    out_h_end = tl.minimum(out_h_end, input_h)
    out_w_end = tl.minimum(out_w_end, input_w)

    # Create the range of input indices for this block
    h_range = tl.arange(0, BLOCK_SIZE)
    w_range = tl.arange(0, BLOCK_SIZE)

    # Compute the corresponding input indices (with padding)
    # For each output position, compute the input indices via stride and padding
    h_indices = (h_range + out_h_start) * stride_h
    w_indices = (w_range + out_w_start) * stride_w

    # Compute the input indices with padding
    # We use the padding to handle the boundaries
    h_indices = h_indices + pad_h
    w_indices = w_indices + pad_w

    # Mask for valid input indices
    h_mask = (h_indices < input_h)
    w_mask = (w_indices < input_w)
    valid_mask = h_mask & w_mask

    # Load input features (batch, in_channels, H, W)
    # We load the input in a tiled fashion
    input_offset = batch_idx * in_channels * input_h * input_w + \
                   tl.arange(0, BLOCK_SIZE) * input_w * input_h + \
                   tl.arange(0, BLOCK_SIZE) * input_h + \
                   tl.arange(0, BLOCK_SIZE)

    # Load input data with masking
    input_data = tl.load(input_ptr + input_offset, mask=valid_mask, other=0.0)

    # Load weights (out_channels, in_channels, 3, 3)
    # We load weights in a tiled way
    weight_offset = tl.arange(0, out_channels) * in_channels * kernel_h * kernel_w + \
                    tl.arange(0, in_channels) * kernel_h * kernel_w + \
                    tl.arange(0, kernel_h) * kernel_w + tl.arange(0, kernel_w)

    # Load weights
    weight_data = tl.load(weight_ptr + weight_offset, mask=valid_mask, other=0.0)

    # Perform convolution: sum over kernel
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Compute the input indices
            h_idx = h_indices[i]
            w_idx = w_indices[j]
            # Compute the input value
            input_val = input_data[i, j]
            # Compute the output value
            for k in range(out_channels):
                for c in range(in_channels):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_in = h_idx + kh
                            w_in = w_idx + kw
                            if h_in < input_h and w_in < input_w:
                                # Load the weight
                                w_val = weight_data[k, c, kh, kw]
                                # Accumulate
                                output[i, j] += input_val * w_val
    # Add bias
    bias_offset = tl.arange(0, out_channels)
    bias_val = tl.load(bias_ptr + bias_offset, mask=bias_offset < out_channels, other=0.0)
    output = output + bias_val

    # Store output
    output_offset = batch_idx * out_channels * out_h_end * out_w_end + \
                    out_h_start * out_w_end + out_w_start
    tl.store(output_ptr + output_offset, output, mask=valid_mask)


@triton.jit
def bn_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    moving_mean_ptr,
    moving_var_ptr,
    output_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a batch of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    # Load moving mean and variance
    mean = tl.load(moving_mean_ptr + offsets, mask=mask, other=0.0)
    var = tl.load(moving_var_ptr + offsets, mask=mask, other=1.0)

    # Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    x_centered = x - mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = x_centered * inv_std * gamma + beta

    # Store output
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    block_id = tl.program_id(0)
    # Compute the block start
    block_start = block_id * BLOCK_SIZE
    # Create offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for valid indices
    mask = offsets < BLOCK_SIZE
    # Load A and B
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    # Perform multiplication
    c = a * b
    # Store
    tl.store(c_ptr + offsets, c, mask=mask)


def triton_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    out_channels: int = None,
):
    """
    Custom Triton kernel for 2D convolution.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert input.shape[1] == weight.shape[1], "Input channels must match weight input channels."
    assert weight.shape[0] == out_channels, "Out channels must match weight output channels."

    batch_size, in_channels, h, w = input.shape
    out_channels = weight.shape[0]
    kernel_h, kernel_w = 3, 3
    output_h = (h + 2 * padding - kernel_h) // stride + 1
    output_w = (w + 2 * padding - kernel_w) // stride + 1

    # Prepare output
    output = torch.empty((batch_size, out_channels, output_h, output_w), dtype=input.dtype, device=input.device)

    # Define grid
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_h + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr(),
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=h,
        input_w=w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride,
        stride_w=stride,
        pad_h=padding,
        pad_w=padding,
        BLOCK_SIZE=128,
    )
    return output


def triton_bn(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    moving_mean: torch.Tensor,
    moving_var: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Custom Triton kernel for Batch Normalization.
    """
    assert input.is_cuda and gamma.is_cuda and beta.is_cuda and moving_mean.is_cuda and moving_var.is_cuda
    assert input.shape[0] == gamma.shape[0]
    assert input.shape[1] == gamma.shape[1]

    batch_size, channels = input.shape[0], input.shape[1]
    output = torch.empty_like(input)

    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    bn_kernel[grid](
        input_ptr=input.data_ptr(),
        gamma_ptr=gamma.data_ptr(),
        beta_ptr=beta.data_ptr(),
        moving_mean_ptr=moving_mean.data_ptr(),
        moving_var_ptr=moving_var.data_ptr(),
        output_ptr=output.data_ptr(),
        N=channels,
        eps=eps,
        BLOCK_SIZE=128,
    )
    return output


def triton_relu(
    input: torch.Tensor,
):
    """
    Custom Triton kernel for ReLU activation.
    """
    assert input.is_cuda
    output = torch.empty_like(input)

    grid = lambda meta: ((input.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=input.numel(),
        BLOCK_SIZE=128,
    )
    return output


def triton_avgpool(
    input: torch.Tensor,
):
    """
    Custom Triton kernel for Adaptive Average Pooling (to (1,1)).
    """
    assert input.is_cuda
    batch_size, channels, h, w = input.shape
    output = torch.empty((batch_size, channels), device=input.device, dtype=input.dtype)

    # We directly reduce to (1,1)
    # Using a simple reduction kernel
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Simple reduction
    # Each block reduces a batch element
    @triton.jit
    def avgpool_kernel(
        input_ptr,
        output_ptr,
        batch_size: tl.constexpr,
        channels: tl.constexpr,
        h: tl.constexpr,
        w: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        channel_idx = tl.program_id(1)

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < channels

        # Load input
        input_vals = tl.load(input_ptr + batch_idx * channels * h * w + channel_idx * h * w + offsets, mask=mask, other=0.0)
        # Average
        avg = tl.sum(input_vals) / (h * w)
        # Store
        tl.store(output_ptr + batch_idx * channels + channel_idx, avg, mask=mask)

    avgpool_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        channels=channels,
        h=h,
        w=w,
        BLOCK_SIZE=128,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x