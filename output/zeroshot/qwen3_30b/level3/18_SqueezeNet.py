import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width, input_channels, output_channels,
    kernel_size, stride, padding, 
    batch_size,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Block indices
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate block offsets
    block_start_h = pid_h * BLOCK_SIZE_H
    block_start_w = pid_w * BLOCK_SIZE_W
    block_start_c = pid_c * BLOCK_SIZE_C

    # Create indices
    h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
    w = block_start_w + tl.arange(0, BLOCK_SIZE_W)
    c = block_start_c + tl.arange(0, BLOCK_SIZE_C)

    # Mask for out-of-bounds
    h_mask = h < input_height
    w_mask = w < input_width
    c_mask = c < input_channels

    # Calculate input strides
    input_stride_h = input_width * input_channels
    input_stride_w = input_channels
    input_stride_c = 1

    # Output stride
    output_stride_h = input_width * output_channels
    output_stride_w = output_channels
    output_stride_c = 1

    # Weight strides
    weight_stride_h = input_channels * kernel_size * kernel_size
    weight_stride_w = input_channels * kernel_size
    weight_stride_c = input_channels

    # Load input and weight
    input_offsets = (tl.broadcast_to(h[:, None], (BLOCK_SIZE_H, BLOCK_SIZE_W)) * input_stride_h +
                     tl.broadcast_to(w[None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W)) * input_stride_w +
                     tl.broadcast_to(c[None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * input_stride_c)
    input_ptr_offsets = input_ptr + input_offsets
    input_vals = tl.load(input_ptr_offsets, mask=(h_mask[:, None] & w_mask[None, :])[:, :, None] & c_mask[None, None, :], other=0.0)

    # Load weights
    weight_offsets = (tl.broadcast_to(c[:, None, None], (BLOCK_SIZE_C, kernel_size, kernel_size)) * weight_stride_c +
                      tl.broadcast_to(tl.arange(0, kernel_size)[:, None], (BLOCK_SIZE_C, kernel_size, kernel_size)) * weight_stride_w +
                      tl.broadcast_to(tl.arange(0, kernel_size)[None, :], (BLOCK_SIZE_C, kernel_size, kernel_size)) * weight_stride_h)
    weight_ptr_offsets = weight_ptr + weight_offsets
    weight_vals = tl.load(weight_ptr_offsets, mask=(c_mask[:, None, None] & (tl.arange(0, kernel_size)[:, None, None] < kernel_size) & (tl.arange(0, kernel_size)[None, :] < kernel_size)), other=0.0)

    # Convolution computation
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Input indices
            input_h = h - padding + i
            input_w = w - padding + j

            # Clamp to valid input bounds
            valid_h = (input_h >= 0) & (input_h < input_height)
            valid_w = (input_w >= 0) & (input_w < input_width)

            # Load input values for this kernel position
            input_offset = (input_h[:, None] * input_stride_h +
                            input_w[None, :] * input_stride_w +
                            c[None, None, :] * input_stride_c)
            input_offset = input_offset + (tl.broadcast_to(valid_h[:, None], (BLOCK_SIZE_H, BLOCK_SIZE_W)) & tl.broadcast_to(valid_w[None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W)))[:, :, None]
            input_val = tl.load(input_ptr + input_offset, mask=valid_h[:, None] & valid_w[None, :] & c_mask[None, None, :], other=0.0)
            # Compute dot product
            output += input_val[:, :, None] * weight_vals[:, i, j, None]

    # Sum over input channels and kernel size
    output = tl.sum(output, axis=2)

    # Output stride
    output_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    output_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    output_c = tl.arange(0, BLOCK_SIZE_C)
    output_offsets = (tl.broadcast_to(output_h[:, None], (BLOCK_SIZE_H, BLOCK_SIZE_W)) * output_stride_h +
                      tl.broadcast_to(output_w[None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W)) * output_stride_w +
                      tl.broadcast_to(output_c[None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * output_stride_c)
    output_ptr_offsets = output_ptr + output_offsets
    tl.store(output_ptr_offsets, output, mask=(h_mask[:, None] & w_mask[None, :])[:, :, None] & c_mask[None, None, :])

    # Handle bias if needed
    if USE_BIAS:
        bias_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
        bias_ptr_offsets = bias_ptr + bias_offsets
        bias_vals = tl.load(bias_ptr_offsets, mask=c_mask, other=0.0)
        output += bias_vals[None, None, :]


@triton.jit
def relu_kernel(
    input_ptr, output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Block indices
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load input
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output = tl.maximum(input, 0.0)

    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def cat_kernel(
    input1_ptr, input2_ptr, output_ptr,
    batch_size, height, width, ch1, ch2,
    BLOCK_SIZE: tl.constexpr
):
    # Block indices
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_b = tl.program_id(3)

    # Calculate block offsets
    block_start_c = pid_c * BLOCK_SIZE
    block_start_h = pid_h * BLOCK_SIZE
    block_start_w = pid_w * BLOCK_SIZE
    block_start_b = pid_b * BLOCK_SIZE

    # Create indices
    c = block_start_c + tl.arange(0, BLOCK_SIZE)
    h = block_start_h + tl.arange(0, BLOCK_SIZE)
    w = block_start_w + tl.arange(0, BLOCK_SIZE)
    b = block_start_b + tl.arange(0, BLOCK_SIZE)

    # Masks
    c_mask = c < ch1
    h_mask = h < height
    w_mask = w < width
    b_mask = b < batch_size

    # Flatten offsets
    input1_offset = (tl.broadcast_to(b[:, None, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (height * width * ch1) +
                     tl.broadcast_to(h[None, :, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (width * ch1) +
                     tl.broadcast_to(w[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * ch1 +
                     tl.broadcast_to(c[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)))
    input2_offset = (tl.broadcast_to(b[:, None, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (height * width * ch2) +
                     tl.broadcast_to(h[None, :, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (width * ch2) +
                     tl.broadcast_to(w[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * ch2 +
                     tl.broadcast_to(c[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)))

    # Load input1
    input1_vals = tl.load(input1_ptr + input1_offset, mask=(b_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :] & c_mask[None, None, :]), other=0.0)

    # Load input2
    input2_vals = tl.load(input2_ptr + input2_offset, mask=(b_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :] & c_mask[None, None, :]), other=0.0)

    # Concatenate along channel dim
    output_offset = (tl.broadcast_to(b[:, None, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (height * width * (ch1 + ch2)) +
                     tl.broadcast_to(h[None, :, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (width * (ch1 + ch2)) +
                     tl.broadcast_to(w[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (ch1 + ch2) +
                     tl.broadcast_to(c[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)))

    # Store
    tl.store(output_ptr + output_offset, tl.concat([input1_vals, input2_vals], dim=2), mask=(b_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :] & c_mask[None, None, :]))


@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    batch_size, input_height, input_width, output_channels,
    BLOCK_SIZE: tl.constexpr
):
    # Block indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_b = tl.program_id(3)

    # Block size
    h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    b = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask
    h_mask = h < input_height
    w_mask = w < input_width
    c_mask = c < output_channels
    b_mask = b < batch_size

    # Input and output strides
    input_stride_h = input_width * output_channels
    input_stride_w = output_channels
    input_stride_c = 1
    output_stride_h = 1
    output_stride_w = 1
    output_stride_c = 1

    # Compute stride
    stride_h = input_height // 1
    stride_w = input_width // 1

    # Input offset
    input_offset = (tl.broadcast_to(b[:, None, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (input_height * input_width * output_channels) +
                    tl.broadcast_to(h[None, :, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (input_width * output_channels) +
                    tl.broadcast_to(w[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * output_channels +
                    tl.broadcast_to(c[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)))
    input_val = tl.load(input_ptr + input_offset, mask=(b_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :] & c_mask[None, None, :]), other=0.0)

    # Compute mean
    output = tl.sum(input_val, axis=0) / (input_height * input_width)

    # Output offset
    output_offset = (tl.broadcast_to(b[:, None, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (1 * 1 * output_channels) +
                     tl.broadcast_to(h[None, :, None], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * (1 * output_channels) +
                     tl.broadcast_to(w[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) * output_channels +
                     tl.broadcast_to(c[None, None, :], (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)))
    tl.store(output_ptr + output_offset, output, mask=(b_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :] & c_mask[None, None, :]))


def triton_conv2d(x, weight, bias=None, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, input_channels, input_height, input_width = x.shape
    output_channels, _, kernel_size, _ = weight.shape

    # Output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    # Allocate output
    out = torch.empty(batch_size, output_channels, output_height, output_width, device=x.device, dtype=x.dtype)

    # Grid setup
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32

    grid = lambda meta: (
        (output_channels + meta['BLOCK_SIZE_C'] - 1) // meta['BLOCK_SIZE_C'],
        (output_height + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (output_width + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W']
    )

    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        input_height, input_width, input_channels, output_channels,
        kernel_size, stride, padding,
        batch_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C,
        USE_BIAS=(bias is not None)
    )

    return out


def triton_relu(x):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: (x.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']

    relu_kernel[grid](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_cat(input1, input2):
    assert input1.is_cuda and input2.is_cuda, "Tensors must be on CUDA."
    input1 = input1.contiguous()
    input2 = input2.contiguous()

    batch_size, ch1, height, width = input1.shape
    _, ch2, _, _ = input2.shape

    out = torch.empty(batch_size, ch1 + ch2, height, width, device=input1.device, dtype=input1.dtype)

    BLOCK_SIZE = 16
    grid = lambda meta: (
        (ch1 + ch2 + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (height + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (width + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (batch_size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
    )

    cat_kernel[grid](input1, input2, out, batch_size, height, width, ch1, ch2, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_adaptive_avg_pool2d(x):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch_size, channels, height, width = x.shape

    out = torch.empty(batch_size, channels, 1, 1, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 16
    grid = lambda meta: (
        (height + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (width + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (channels + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        (batch_size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
    )

    adaptive_avg_pool2d_kernel[grid](x, out, batch_size, height, width, channels, BLOCK_SIZE=BLOCK_SIZE)
    return out


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = lambda x: triton_relu(x)  # Replaced with Triton
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = lambda x: triton_relu(x)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = lambda x: triton_relu(x)
    
    def forward(self, x):
        x = self.squeeze_activation(triton_conv2d(x, self.squeeze.weight, self.squeeze.bias))
        expand1x1_out = self.expand1x1_activation(triton_conv2d(x, self.expand1x1.weight, self.expand1x1.bias))
        expand3x3_out = self.expand3x3_activation(triton_conv2d(x, self.expand3x3.weight, self.expand3x3.bias))
        return triton_cat(expand1x1_out, expand3x3_out)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            lambda x: triton_relu(x),
            lambda x: F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            lambda x: F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            lambda x: F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            lambda x: triton_relu(x),
            lambda x: triton_adaptive_avg_pool2d(x)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)