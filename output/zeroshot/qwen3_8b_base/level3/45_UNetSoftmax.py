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
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    num_channels,  # Number of input channels
    out_channels,  # Number of output channels
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Compute the output position (i, j)
    i = block_idx // (out_channels * out_channels)
    j = (block_idx // out_channels) % out_channels
    k = block_idx % out_channels

    # Compute the input position (n, c, h, w)
    n = i // (input_shape[2] * input_shape[3])
    c = (i // input_shape[3]) % input_shape[2]
    h = i % input_shape[3]
    w = j % input_shape[3]

    # Compute the output position (o_h, o_w)
    o_h = (h + padding[0]) // stride[0]
    o_w = (w + padding[1]) // stride[1]

    # Compute the output offset
    o = o_h * input_shape[3] + o_w
    o = o * out_channels + j

    # Compute the input offset
    i = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + h * input_shape[3] + w

    # Load the weight
    weight = tl.load(weight_ptr + k * num_channels * kernel_size[0] * kernel_size[1] + c * kernel_size[0] * kernel_size[1] + (h - padding[0]) * kernel_size[1] + (w - padding[1]), other=0.0)

    # Load the input
    input_val = tl.load(input_ptr + i, other=0.0)

    # Compute the output
    output = input_val * weight

    # Store the output
    tl.store(output_ptr + o, output)


@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Compute the output position (i, j)
    i = block_idx // (input_shape[2] * input_shape[3])
    j = (block_idx // input_shape[3]) % input_shape[2]
    k = block_idx % input_shape[3]

    # Compute the input offset
    input_offset = i * input_shape[1] * input_shape[2] * input_shape[3] + j * input_shape[2] * input_shape[3] + k

    # Load the input
    input_val = tl.load(input_ptr + input_offset, other=0.0)

    # Compute the softmax
    exp_val = tl.exp(input_val - tl.max(input_val))
    sum_exp = tl.sum(exp_val)
    output_val = exp_val / sum_exp

    # Store the output
    tl.store(output_ptr + input_offset, output_val)


def triton_conv2d(input, weight, input_shape, kernel_size, stride, padding, num_channels, out_channels):
    # Prepare output tensor
    output_shape = (input_shape[0], out_channels, input_shape[2], input_shape[3])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_shape, kernel_size, stride, padding, num_channels, out_channels, BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=1)
    return output


def triton_softmax(input, input_shape):
    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softmax_kernel[grid](input, output, input_shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.double_conv(x)


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