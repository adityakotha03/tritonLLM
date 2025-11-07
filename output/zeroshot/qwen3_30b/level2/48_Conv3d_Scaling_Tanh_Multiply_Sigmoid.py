import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    kernel_size,
    stride,
    padding,
    output_depth,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = pid // (output_height * output_width)
    pid_y = (pid - pid_z * output_height * output_width) // output_width
    pid_x = pid - pid_z * output_height * output_width - pid_y * output_width

    if pid_z >= output_depth or pid_y >= output_height or pid_x >= output_width:
        return

    for oc in tl.static_range(out_channels):
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ic in tl.static_range(in_channels):
            for kz in tl.static_range(kernel_size):
                for ky in tl.static_range(kernel_size):
                    for kx in tl.static_range(kernel_size):
                        iz = pid_z * stride + kz - padding
                        iy = pid_y * stride + ky - padding
                        ix = pid_x * stride + kx - padding

                        if iz < 0 or iz >= depth or iy < 0 or iy >= height or ix < 0 or ix >= width:
                            continue

                        input_offset = (ic * depth * height * width + iz * height * width + iy * width + ix) * BLOCK_SIZE
                        weight_offset = (oc * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kz * kernel_size * kernel_size + ky * kernel_size + kx) * BLOCK_SIZE

                        input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, BLOCK_SIZE) < 1, other=0.0)
                        weight_val = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < 1, other=0.0)
                        acc += input_val * weight_val

        bias_val = tl.load(bias_ptr + oc * BLOCK_SIZE, mask=tl.arange(0, BLOCK_SIZE) < 1, other=0.0)
        output_val = acc + bias_val

        output_offset = (oc * output_depth * output_height * output_width + pid_z * output_height * output_width + pid_y * output_width + pid_x) * BLOCK_SIZE
        tl.store(output_ptr + output_offset, output_val, mask=tl.arange(0, BLOCK_SIZE) < 1)


@triton.jit
def scale_tanh_scale_kernel(
    x_ptr,
    scaling_factor_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    out_channels,
    output_depth,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    oc = pid // (output_depth * output_height * output_width)
    if oc >= out_channels:
        return

    for pid_z in tl.static_range(output_depth):
        for pid_y in tl.static_range(output_height):
            for pid_x in tl.static_range(output_width):
                offset = (oc * output_depth * output_height * output_width + pid_z * output_height * output_width + pid_y * output_width + pid_x) * BLOCK_SIZE

                x_val = tl.load(x_ptr + offset, mask=tl.arange(0, BLOCK_SIZE) < 1, other=0.0)
                scale_val = tl.load(scaling_factor_ptr + oc * BLOCK_SIZE, mask=tl.arange(0, BLOCK_SIZE) < 1, other=0.0)
                x_scaled = x_val * scale_val

                x_tanh = tl.tanh(x_scaled)

                bias_val = tl.load(bias_ptr + oc * BLOCK_SIZE, mask=tl.arange(0, BLOCK_SIZE) < 1, other=0.0)
                x_final = x_tanh * bias_val

                x_sigmoid = 1.0 / (1.0 + tl.exp(-x_final))

                tl.store(out_ptr + offset, x_sigmoid, mask=tl.arange(0, BLOCK_SIZE) < 1)


def triton_conv3d(x, weight, bias, stride=1, padding=1):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    output_depth = (depth + 2 * padding - kernel_size) // stride + 1
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1

    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)

    grid = lambda meta: (out_channels * output_depth * output_height * output_width,)
    conv3d_kernel[grid](x, weight, output, bias, batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding, output_depth, output_height, output_width, BLOCK_SIZE=1)

    return output


def triton_scale_tanh_scale(x, scaling_factor, bias):
    batch_size, out_channels, depth, height, width = x.shape

    output = torch.empty_like(x)

    grid = lambda meta: (out_channels * depth * height * width,)
    scale_tanh_scale_kernel[grid](x, scaling_factor, bias, output, batch_size, out_channels, depth, height, width, BLOCK_SIZE=1)

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=True)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = triton_conv3d(x, self.conv.weight, self.conv.bias, stride=1, padding=1)
        x = triton_scale_tanh_scale(x, self.scaling_factor, self.bias)
        return x