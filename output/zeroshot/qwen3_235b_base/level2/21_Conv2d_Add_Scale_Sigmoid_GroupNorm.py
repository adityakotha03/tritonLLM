import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_bias_scale_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, scale_ptr, output_ptr,
    batch_size, out_channels, out_height, out_width, in_channels, in_height, in_width,
    kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w,
    eps, num_groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_gh = tl.program_id(1)
    pid_gw = tl.program_id(2)
    pid_cg = tl.program_id(3)

    # Compute group and channel
    group_id = pid_cg // (out_channels // num_groups)
    channel_group_id = pid_cg % (out_channels // num_groups)
    c = group_id * (out_channels // num_groups) + channel_group_id

    # Output spatial dimensions
    out_h = pid_gh
    out_w = pid_gw

    # Input image is padded
    input_hw = in_height * in_width
    input_chw = in_channels * input_hw
    weight_hkw = kernel_size_h * kernel_size_w
    weight_chwk = in_channels * weight_hkw
    output_hw = out_height * out_width
    output_chw = out_channels * output_hw

    # Pointers to input batch
    input_batch_offset = pid_b * input_chw
    output_batch_offset = pid_b * output_chw

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over input channels and kernel
    for k in range(0, in_channels * kernel_size_h * kernel_size_w, BLOCK_SIZE_K):
        # Current block of input channels
        kc = k // weight_hkw
        kh = (k % weight_hkw) // kernel_size_w
        kw = (k % kernel_size_w)

        # Input pixel coordinates
        ih = out_h * stride_h - padding_h + kh
        iw = out_w * stride_w - padding_w + kw

        # Bounds check for input
        valid_h = (ih >= 0) and (ih < in_height)
        valid_w = (iw >= 0) and (iw < in_width)
        valid_c = (kc < in_channels)

        # Mask for input access
        mask_input = valid_h and valid_w and valid_c
        input_offset = input_batch_offset + kc * input_hw + ih * in_width + iw
        input_val = tl.load(input_ptr + input_offset, mask=mask_input, other=0.0)

        # Weight offset
        weight_offset = c * weight_chwk + kc * weight_hkw + kh * kernel_size_w + kw
        weight_val = tl.load(weight_ptr + weight_offset)

        # Accumulate
        acc += input_val.to(tl.float32) * weight_val.to(tl.float32)

    # Store result in temporary buffer (bias + scale + sigmoid)
    acc = acc + tl.load(bias_ptr + c)
    acc = acc * tl.load(scale_ptr + c)
    acc = tl.sigmoid(acc)

    # Write back to output
    output_offset = output_batch_offset + c * output_hw + out_h * out_width + out_w
    tl.store(output_ptr + output_offset, acc)


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    N, C, HxW,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = C // num_groups

    # Stride
    offset_ng = pid_n * C * HxW + pid_g * group_size * HxW
    group_mask = tl.arange(0, BLOCK_SIZE) < group_size * HxW

    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute mean
    for i in range(0, group_size * HxW, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = group_mask and (offsets < group_size * HxW)
        x = tl.load(x_ptr + offset_ng + offsets, mask=mask, other=0.0).to(tl.float32)
        mean += x
        var += x * x

    mean = tl.sum(mean) / (group_size * HxW)
    var = tl.sum(var) / (group_size * HxW) - mean * mean

    # Normalize and apply affine
    for i in range(0, group_size * HxW, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = group_mask and (offsets < group_size * HxW)
        x = tl.load(x_ptr + offset_ng + offsets, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) / tl.sqrt(var + eps)

        c_local = offsets % group_size
        gamma = tl.load(gamma_ptr + pid_g * group_size + c_local, mask=mask % group_size < group_size, other=1.0)
        beta = tl.load(beta_ptr + pid_g * group_size + c_local, mask=mask % group_size < group_size, other=0.0)

        output = gamma * x_hat + beta
        tl.store(y_ptr + offset_ng + offsets, output, mask=mask)


def fused_conv_bias_scale_sigmoid(input, weight, bias, scale, stride, padding):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape

    out_height = (in_height + 2 * padding[0] - kernel_size_h) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_size_w) // stride[1] + 1

    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)

    # Launch kernel
    def grid(META):
        return (batch_size, out_height, out_width, out_channels)

    fused_conv_bias_scale_sigmoid_kernel[grid](
        input, weight, bias, scale, output,
        batch_size, out_channels, out_height, out_width,
        in_channels, in_height, in_width,
        kernel_size_h, kernel_size_w, stride[0], stride[1], padding[0], padding[1],
        1e-5, 1,  # dummy eps and num_groups for now
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
    )
    return output


def triton_group_norm(x, num_groups, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    HxW = H * W
    y = torch.empty_like(x)

    def grid(META):
        return (N, num_groups)

    group_norm_kernel[grid](
        x, weight, bias, y,
        N, C, HxW,
        num_groups=num_groups,
        eps=eps,
        BLOCK_SIZE=1024
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model using fused Triton kernels for convolution + bias + scale + sigmoid,
    followed by Triton-implemented group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # Fused convolution, bias, scale, sigmoid using Triton
        x = fused_conv_bias_scale_sigmoid(
            x, self.conv.weight, self.bias.view(-1), self.scale.view(-1),
            stride=self.conv.stride, padding=self.conv.padding
        )
        # Group normalization using Triton
        x = triton_group_norm(x, self.group_norm.num_groups, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps)
        return x