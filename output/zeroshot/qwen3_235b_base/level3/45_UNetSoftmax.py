import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_rows, row_stride, BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = input_ptr + row_id * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < row_stride
    row = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start = output_ptr + row_id * row_stride
    tl.store(output_row_start + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    shape = x.shape
    dim = dim % x.ndim  # handle negative dim
    row_stride = shape[dim]
    n_rows = n_elements // row_stride
    out = torch.empty_like(x)
    # Choose block size as power of 2 >= row_stride
    BLOCK_SIZE = 1
    while BLOCK_SIZE < row_stride:
        BLOCK_SIZE *= 2
    # Clamp to reasonable maximum
    BLOCK_SIZE = min(max(BLOCK_SIZE, 128), 4096)
    grid = lambda meta: (n_rows,)
    softmax_kernel[grid](
        x, out,
        n_rows, row_stride,
        BLOCK_SIZE=tl.constexpr(BLOCK_SIZE)
    )
    return out


@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr,
    weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr,
    eps,
    n_elements, num_channels, channel_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    channel_ids = (offsets // channel_size) % num_channels
    mean = tl.load(running_mean_ptr + channel_ids, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + channel_ids, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_ids, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + channel_ids, mask=mask, other=0.0)
    bn_output = (input_vals - mean) / tl.sqrt(var + eps) * weight + bias
    tl.store(output_ptr + offsets, bn_output, mask=mask)


def triton_batch_norm(
    x: torch.Tensor,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float
):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and running_mean.is_cuda and running_var.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    num_channels = x.shape[1]
    channel_size = x.shape[2] * x.shape[3] if x.ndim == 4 else x.shape[1] if x.ndim == 2 else 1
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    batch_norm_kernel[grid](
        x, out,
        weight, bias,
        running_mean, running_var,
        eps,
        n_elements, num_channels, channel_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class TritonBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5

    def forward(self, x):
        if self.training:
            batch_mean = x.mean([0, 2, 3])
            batch_var = x.var([0, 2, 3], unbiased=False)
            self.running_mean.copy_(0.9 * self.running_mean + 0.1 * batch_mean)
            self.running_var.copy_(0.9 * self.running_var + 0.1 * batch_var)
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        return triton_batch_norm(
            x, self.weight, self.bias,
            self.running_mean, self.running_var, self.eps
        )


class TritonSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_softmax(x, self.dim)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            TritonBatchNorm2d(out_channels),
            TritonSoftmax(dim=-1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            TritonBatchNorm2d(out_channels),
            TritonSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.double_conv(x)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(ModelNew, self).__init__()
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