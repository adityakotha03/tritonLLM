import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, depth, height, width)
    weight_shape,  # (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
    output_shape,  # (batch, out_channels, depth, height, width)
    stride_d, stride_h, stride_w,  # Strides for transposed convolution
    padding_d, padding_h, padding_w,  # Padding for transposed convolution
    dilation_d, dilation_h, dilation_w,  # Dilation for transposed convolution
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Get thread ID within the block
    tid = tl.program_id(1)
    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tid
    # Get output dimensions
    batch, out_channels, out_depth, out_height, out_width = output_shape
    # Compute the input dimensions
    in_channels = input_shape[1]
    # Compute the kernel dimensions
    kernel_depth, kernel_height, kernel_width = weight_shape[2], weight_shape[3], weight_shape[4]
    # Compute the output index in 3D
    out_d = out_idx // (out_height * out_width)
    out_h = (out_idx // out_width) % out_height
    out_w = out_idx % out_width
    # Compute the input indices for each channel
    for c in range(out_channels // groups):
        # Get the input channel index
        in_c = c * groups
        # Compute the input indices
        in_d_start = (out_d - 1) * stride_d + padding_d
        in_h_start = (out_h - 1) * stride_h + padding_h
        in_w_start = (out_w - 1) * stride_w + padding_w
        # Compute the input indices for each kernel position
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    in_d = in_d_start + kd * dilation_d
                    in_h = in_h_start + kh * dilation_h
                    in_w = in_w_start + kw * dilation_w
                    # Check if the input index is valid
                    if in_d < 0 or in_d >= input_shape[2] or in_h < 0 or in_h >= input_shape[3] or in_w < 0 or in_w >= input_shape[4]:
                        continue
                    # Compute the input index
                    in_idx = in_c + in_channels * (in_d * input_shape[3] * input_shape[4] + in_h * input_shape[4] + in_w)
                    # Compute the weight index
                    weight_idx = c + out_channels * (kd * kernel_height * kernel_width + kh * kernel_width + kw)
                    # Load input and weight
                    input_val = tl.load(input_ptr + in_idx, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < out_channels * in_channels * kernel_depth * kernel_height * kernel_width, other=0.0)
                    # Accumulate the result
                    output_val = tl.load(output_ptr + out_idx, mask=out_idx < batch * out_channels * out_depth * out_height * out_width, other=0.0)
                    output_val += input_val * weight_val
                    tl.store(output_ptr + out_idx, output_val, mask=out_idx < batch * out_channels * out_depth * out_height * out_width)

def triton_conv_transpose3d(input, weight, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, groups):
    """
    Triton implementation of 3D transposed convolution.
    """
    batch, in_channels, in_depth, in_height, in_width = input.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    out_depth = (in_depth - 1) * stride_d + kernel_depth - 2 * padding_d
    out_height = (in_height - 1) * stride_h + kernel_height - 2 * padding_h
    out_width = (in_width - 1) * stride_w + kernel_width - 2 * padding_w
    output = torch.zeros((batch, out_channels, out_depth, out_height, out_width), device=input.device, dtype=input.dtype)
    # Define block size
    BLOCK_SIZE = 1024
    # Determine grid size
    grid = lambda meta: ( (batch * out_channels * out_depth * out_height * out_width + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    # Launch the kernel
    conv_transpose3d_kernel[grid](input, weight, output, input.shape, weight.shape, output.shape, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, groups, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def batch_norm3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    mean_ptr,  # Pointer to mean tensor
    rstd_ptr,  # Pointer to reciprocal standard deviation tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    out_idx = pid * BLOCK_SIZE + tid
    batch, channels, depth, height, width = input_shape
    for b in range(batch):
        for c in range(channels):
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        idx = b * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w
                        input_val = tl.load(input_ptr + idx, mask=idx < batch * channels * depth * height * width, other=0.0)
                        weight_val = tl.load(weight_ptr + c, mask=c < channels, other=0.0)
                        bias_val = tl.load(bias_ptr + c, mask=c < channels, other=0.0)
                        mean_val = tl.load(mean_ptr + c, mask=c < channels, other=0.0)
                        rstd_val = tl.load(rstd_ptr + c, mask=c < channels, other=0.0)
                        output_val = (input_val - mean_val) * rstd_val * weight_val + bias_val
                        tl.store(output_ptr + idx, output_val, mask=idx < batch * channels * depth * height * width)

def triton_batch_norm3d(input, weight, bias, mean, rstd):
    """
    Triton implementation of 3D batch normalization.
    """
    batch, channels, depth, height, width = input.shape
    output = torch.zeros_like(input)
    # Define block size
    BLOCK_SIZE = 1024
    # Determine grid size
    grid = lambda meta: ( (batch * channels * depth * height * width + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    # Launch the kernel
    batch_norm3d_kernel[grid](input, weight, bias, mean, rstd, output, input.shape, eps=1e-5, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def global_avg_pool3d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    out_idx = pid * BLOCK_SIZE + tid
    batch, channels, depth, height, width = input_shape
    for b in range(batch):
        for c in range(channels):
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        idx = b * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w
                        input_val = tl.load(input_ptr + idx, mask=idx < batch * channels * depth * height * width, other=0.0)
                        # Compute average
                        avg = input_val / (depth * height * width)
                        tl.store(output_ptr + out_idx, avg, mask=out_idx < batch * channels)

def triton_global_avg_pool3d(input):
    """
    Triton implementation of 3D global average pooling.
    """
    batch, channels, depth, height, width = input.shape
    output = torch.zeros((batch, channels), device=input.device, dtype=input.dtype)
    # Define block size
    BLOCK_SIZE = 1024
    # Determine grid size
    grid = lambda meta: ( (batch * channels + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    # Launch the kernel
    global_avg_pool3d_kernel[grid](input, output, input.shape, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.ones(out_channels))
        self.shift = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Triton-based transposed convolution
        kernel_size = self.kernel_size
        stride_d = self.scale_factor
        stride_h = self.scale_factor
        stride_w = self.scale_factor
        padding_d = (kernel_size[0] - 1) // 2
        padding_h = (kernel_size[1] - 1) // 2
        padding_w = (kernel_size[2] - 1) // 2
        dilation_d = 1
        dilation_h = 1
        dilation_w = 1
        groups = 1
        x = triton_conv_transpose3d(x, self.weight, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, groups)
        x = x * self.scale_factor
        # Triton-based batch normalization
        x = triton_batch_norm3d(x, self.weight, self.bias, self.running_mean, self.running_var)
        # Triton-based global average pooling
        x = triton_global_avg_pool3d(x)
        return x