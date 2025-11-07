import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to kernel weights
    bias_ptr,  # Pointer to bias (if exists)
    output_ptr,  # Pointer to output tensor
    input_stride0, input_stride1, input_stride2, input_stride3,  # Strides for input
    weight_stride0, weight_stride1, weight_stride2, weight_stride3,  # Strides for weights
    bias_stride,  # Stride for bias
    output_stride0, output_stride1, output_stride2, output_stride3,  # Strides for output
    batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Calculate thread indices
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Determine output region this block processes
    h_start = pid_h * BLOCK_H - padding
    w_start = pid_w * BLOCK_W - padding

    # Output spatial dimensions
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    # Ensure we don't go out of bounds
    if pid_h >= out_h or pid_w >= out_w:
        return

    # Load output tile
    offs_h = tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = h_start + offs_h
    offs_w = w_start + offs_w

    # Mask out-of-bounds spatial indices
    mask_h = offs_h < height
    mask_w = offs_w < width
    mask_hw = mask_h[:, None] & mask_w[None, :]
    mask_c = offs_c < in_channels

    # Load input values
    input_ptrs = input_ptr + (
        (pid_c * out_channels) * input_stride0 +
        offs_h[:, None, None] * input_stride1 +
        offs_w[None, :, None] * input_stride2 +
        offs_c[None, None, :] * input_stride3
    )
    input_vals = tl.load(input_ptrs, mask=mask_hw[:, :, None] & mask_c[None, None, :], other=0.0)

    # Load kernel weights
    weight_ptrs = weight_ptr + (
        pid_c * out_channels * kernel_size * kernel_size * in_channels +
        offs_h[None, None, :, None] * weight_stride1 +
        offs_w[None, None, None, :] * weight_stride2 +
        offs_c[None, :, None, None] * weight_stride3
    )
    weight_vals = tl.load(weight_ptrs, mask=mask_hw[:, :, None, None] & mask_c[None, None, None, :], other=0.0)

    # Perform convolution: sum over spatial and channel dims
    # Output: [BLOCK_H, BLOCK_W, BLOCK_C]
    # Result: [BLOCK_H, BLOCK_W]
    output = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            for c in range(in_channels):
                output += input_vals[i, j, c] * weight_vals[i, j, c]

    # Apply bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c * out_channels + offs_c, mask=mask_c, other=0.0)
        output = output + bias[None, None, :]

    # Store output
    output_ptrs = output_ptr + (
        (pid_c * out_channels) * output_stride0 +
        offs_h[:, None, None] * output_stride1 +
        offs_w[None, :, None] * output_stride2 +
        offs_c[None, None, :] * output_stride3
    )
    tl.store(output_ptrs, output, mask=mask_hw[:, :, None] & mask_c[None, None, :])


@triton.jit
def batch_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    batch_size, channels, height, width,
    eps: tl.float32,
    BLOCK_C: tl.constexpr
):
    # Each block handles one channel
    pid_c = tl.program_id(0)

    # Calculate offset in output
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    # Load mean and variance
    mean = tl.load(mean_ptr + offs_c, mask=mask_c, other=0.0)
    var = tl.load(var_ptr + offs_c, mask=mask_c, other=0.0)

    # Load weight and bias
    weight = tl.load(weight_ptr + offs_c, mask=mask_c, other=0.0)
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    # Compute stride
    stride_c = channels
    stride_h = stride_c * height
    stride_w = stride_h * width

    # Load input
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < channels
    x_ptrs = x_ptr + (tl.arange(0, batch_size)[:, None] * stride_c + offs_c[None, :])
    x_vals = tl.load(x_ptrs, mask=mask[None, :], other=0.0)  # [batch, channels]

    # Normalize
    x_norm = (x_vals - mean[None, :]) / tl.sqrt(var[None, :] + eps)
    out = x_norm * weight[None, :] + bias[None, :]

    # Store output
    out_ptrs = out_ptr + (tl.arange(0, batch_size)[:, None] * stride_c + offs_c[None, :])
    tl.store(out_ptrs, out, mask=mask[None, :])


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes one row of the flattened input
    pid = tl.program_id(0)
    pid_c = pid % channels
    pid_b = pid // channels

    # Flattened spatial size
    spatial_size = height * width
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < spatial_size

    # Input and output pointers
    input_ptrs = input_ptr + pid_b * channels * spatial_size + pid_c * spatial_size + offs
    output_ptrs = output_ptr + pid_b * channels * spatial_size + pid_c * spatial_size + offs

    # Load input
    x = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    # Compute exp
    x_exp = tl.exp(x)
    # Compute sum
    x_sum = tl.sum(x_exp, axis=0)
    # Normalize
    output = x_exp / x_sum
    # Store
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def cat_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    batch_size, in_channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes a channel of a spatial location
    pid = tl.program_id(0)
    c = pid % in_channels
    h = (pid // in_channels) % height
    w = (pid // in_channels) // height
    b = pid // (in_channels * height * width)

    # Compute offsets
    offs_c = c
    offs_h = h
    offs_w = w
    offs_b = b

    # Compute pointers
    ptr1 = x1_ptr + (offs_b * in_channels + offs_c) * height * width + offs_h * width + offs_w
    ptr2 = x2_ptr + (offs_b * in_channels + offs_c) * height * width + offs_h * width + offs_w
    out_ptr = out_ptr + (offs_b * (in_channels * 2) + offs_c) * height * width + offs_h * width + offs_w

    x1 = tl.load(ptr1, mask=True, other=0.0)
    x2 = tl.load(ptr2, mask=True, other=0.0)
    tl.store(out_ptr, x1)
    tl.store(out_ptr + (in_channels * height * width), x2)


class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        out_h = (height + 2 * padding - kernel_size) // stride + 1
        out_w = (width + 2 * padding - kernel_size) // stride + 1

        # Prepare output
        out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

        # Compute strides
        input_stride0, input_stride1, input_stride2, input_stride3 = x.stride()
        weight_stride0, weight_stride1, weight_stride2, weight_stride3 = self.weight.stride()
        bias_stride = self.bias.stride()[0] if self.bias is not None else 0
        output_stride0, output_stride1, output_stride2, output_stride3 = out.stride()

        # Set kernel config
        BLOCK_H = 16
        BLOCK_W = 16
        BLOCK_C = 32

        # Grid
        grid = lambda meta: (out_channels, (out_h + meta['BLOCK_H'] - 1) // meta['BLOCK_H'],
                             (out_w + meta['BLOCK_W'] - 1) // meta['BLOCK_W'])

        # Launch kernel
        conv2d_kernel[grid](
            x, self.weight, self.bias, out,
            input_stride0, input_stride1, input_stride2, input_stride3,
            weight_stride0, weight_stride1, weight_stride2, weight_stride3,
            bias_stride,
            output_stride0, output_stride1, output_stride2, output_stride3,
            batch_size, in_channels, out_channels, height, width,
            kernel_size, stride, padding,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
        )

        return out


class TritonBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = torch.empty_like(x)

        # Launch kernel
        BLOCK_C = 32
        grid = lambda meta: (channels,)

        batch_norm_kernel[grid](
            x, self.weight, self.bias, self.running_mean, self.running_var, out,
            batch_size, channels, height, width, self.eps, BLOCK_C=BLOCK_C
        )

        return out


class TritonSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        spatial_size = height * width
        out = torch.empty_like(x)

        # Grid for spatial dimension
        grid = lambda meta: ((batch_size * channels + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        BLOCK_SIZE = 32

        softmax_kernel[grid](
            x, out, batch_size, channels, height, width, BLOCK_SIZE=BLOCK_SIZE
        )

        return out


class TritonAdd(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out


class TritonCat(nn.Module):
    def forward(self, x1, x2):
        batch_size, channels, height, width = x1.shape
        out = torch.empty(batch_size, channels * 2, height, width, device=x1.device, dtype=x1.dtype)

        BLOCK_SIZE = 32
        grid = lambda meta: ((batch_size * 2 * channels * height * width + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

        cat_kernel[grid](x1, x2, out, batch_size, channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
        return out


# Modified DoubleConv using Triton kernels
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = TritonConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = TritonBatchNorm2d(out_channels)
        self.act1 = TritonSoftmax(dim=1)
        self.conv2 = TritonConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = TritonBatchNorm2d(out_channels)
        self.act2 = TritonSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


# Optimized U-Net Model
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)