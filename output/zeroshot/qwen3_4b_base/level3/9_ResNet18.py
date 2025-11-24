import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (B, C_in, H, W)
    weight_ptr,  # pointer to weight tensor (C_out, C_in, 3, 3)
    bias_ptr,  # pointer to bias tensor (C_out,)
    output_ptr,  # pointer to output tensor (B, C_out, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get the program ID for the block
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Compute the block's spatial coordinates
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # Define the block size for each dimension
    h_end = min(h_start + BLOCK_SIZE_H, input_height)
    w_end = min(w_start + BLOCK_SIZE_W, input_width)

    # Compute the output dimensions
    h_out = (input_height + 2 * padding - kernel_size) // stride + 1
    w_out = (input_width + 2 * padding - kernel_size) // stride + 1

    # Define the range of output positions
    h_range = tl.arange(0, BLOCK_SIZE_H)
    w_range = tl.arange(0, BLOCK_SIZE_W)

    # Compute the output position for this block
    h_idx = h_start + h_range
    w_idx = w_start + w_range

    # Compute the corresponding input positions with padding
    # For each output position, compute input positions via convolution
    # Input: (B, C_in, H, W), Output: (B, C_out, H_out, W_out)
    # We process each output position and compute convolution over kernel
    # We use a loop over output channels and input channels
    # For each output channel, we compute the convolution with all input channels

    # Define the output channel loop
    for out_c in tl.arange(0, out_channels):
        # Load weight for this output channel
        weights = tl.load(weight_ptr + out_c * in_channels * kernel_size * kernel_size,
                          shape=(in_channels, kernel_size, kernel_size),
                          mask=(out_c < out_channels))

        # Initialize output accumulator
        out = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

        # Loop over input channels
        for in_c in tl.arange(0, in_channels):
            # Load input data for this channel
            # Input is (B, C_in, H, W), so we need to index properly
            # We use the current output position to compute input position
            # with padding and stride
            # Input position: (h_idx + dh, w_idx + dw) with dh, dw in [-1, 1]
            # We loop over kernel positions
            kernel_h = tl.arange(0, kernel_size)
            kernel_w = tl.arange(0, kernel_size)

            # Compute input indices with padding
            # h_in = h_idx + kernel_h - padding
            # w_in = w_idx + kernel_w - padding
            # But we need to mask out-of-bounds
            h_in = h_idx + kernel_h - padding
            w_in = w_idx + kernel_w - padding

            # Mask for valid indices
            h_mask = (h_in >= 0) & (h_in < input_height)
            w_mask = (w_in >= 0) & (w_in < input_width)

            # Load input values
            # Input: (B, C_in, H, W) -> we need to index batch, channel, h_in, w_in
            # We use the current output position (h_idx, w_idx) and input channel
            # and load from input tensor
            # Input pointer: (batch, in_c, h_in, w_in)
            # We use tl.load with mask
            input_val = tl.load(
                input_ptr + (batch_size * in_channels * input_height * input_width + 
                             batch_size * in_c * input_height * input_width + 
                             h_in * input_width + w_in),
                mask=(h_mask & w_mask),
                other=0.0
            )

            # Load weight for this input channel
            weight_val = tl.load(
                weight_ptr + out_c * in_channels * kernel_size * kernel_size + in_c * kernel_size * kernel_size + 
                kernel_h * kernel_size + kernel_w,
                mask=(kernel_h < kernel_size) & (kernel_w < kernel_size),
                other=0.0
            )

            # Accumulate the convolution
            out += input_val * weight_val

        # Store output
        # Output: (B, C_out, H_out, W_out)
        # We store at (h_idx, w_idx) for this output channel
        output_idx = out_c * h_out * w_out + h_idx * w_out + w_idx
        tl.store(output_ptr + output_idx, out, mask=(h_idx < h_end) & (w_idx < w_end))


@triton.jit
def batch_norm_kernel(
    x_ptr,  # input tensor (B, C, H, W)
    gamma_ptr,  # gamma (C,)
    beta_ptr,  # beta (C,)
    running_mean_ptr,  # running_mean (C,)
    running_var_ptr,  # running_var (C,)
    eps: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load gamma and beta
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)

    # Load running mean and variance
    mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + offsets, mask=mask, other=1.0)

    # Compute batch norm
    # x = (x - mean) / sqrt(var + eps) * gamma + beta
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * gamma + beta

    # Store output
    tl.store(x_ptr + offsets, out, mask=mask)


@triton.jit
def relu_kernel(
    x_ptr,  # input tensor (B, C, H, W)
    out_ptr,  # output tensor (B, C, H, W)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def avgpool_kernel(
    x_ptr,  # input tensor (B, C, H, W)
    out_ptr,  # output tensor (B, C, 1, 1)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Average over spatial dimensions
    out = x.sum() / (H * W)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def linear_kernel(
    x_ptr,  # input tensor (B, in_features)
    w_ptr,  # weight tensor (out_features, in_features)
    b_ptr,  # bias tensor (out_features,)
    out_ptr,  # output tensor (B, out_features)
    B: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_features

    # Load weights and bias
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Matrix multiply
    out = tl.dot(x, w) + b
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    output_padding: int = 0,
    groups: int = 1,
):
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert input.dim() == 4 and weight.dim() == 4, "Input and weight must be 4D tensors"
    assert input.shape[1] == weight.shape[1], "Input channels must match weight input channels"

    batch_size, in_channels, H, W = input.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Output dimensions
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    output = torch.empty((batch_size, out_channels, H_out, W_out), dtype=input.dtype, device=input.device)

    # Define grid
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = lambda meta: (
        (H_out + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    conv2d_kernel[
        grid,
        {
            "BLOCK_SIZE_H": BLOCK_SIZE_H,
            "BLOCK_SIZE_W": BLOCK_SIZE_W,
        }
    ](
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        H,
        W,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
    return output


def triton_batch_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
):
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and running_mean.is_cuda and running_var.is_cuda
    assert x.dim() == 4 and gamma.dim() == 1 and beta.dim() == 1 and running_mean.dim() == 1 and running_var.dim() == 1

    B, C, H, W = x.shape
    output = torch.empty_like(x)

    # Use the same block size for all channels
    BLOCK_SIZE = 128
    grid = lambda meta: ((C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    batch_norm_kernel[
        grid,
        {"BLOCK_SIZE": BLOCK_SIZE}
    ](
        x.data_ptr(),
        gamma.data_ptr(),
        beta.data_ptr(),
        running_mean.data_ptr(),
        running_var.data_ptr(),
        eps,
        B,
        C,
        H,
        W,
        BLOCK_SIZE,
    )
    return output


def triton_relu(x: torch.Tensor):
    assert x.is_cuda
    output = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        BLOCK_SIZE,
    )
    return output


def triton_avgpool(x: torch.Tensor):
    assert x.is_cuda
    output = torch.empty((x.shape[0], x.shape[1], 1, 1), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 128
    grid = lambda meta: ((x.shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    avgpool_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        BLOCK_SIZE,
    )
    return output


def triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    assert x.dim() == 2 and w.dim() == 2 and b.dim() == 1
    output = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 128
    grid = lambda meta: ((w.shape[0] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    linear_kernel[grid](
        x.data_ptr(),
        w.data_ptr(),
        b.data_ptr(),
        output.data_ptr(),
        x.shape[0],
        x.shape[1],
        w.shape[0],
        BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

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