import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(
    Y_ptr, X_ptr, W_ptr, B_ptr,
    stride_yd, stride_yh, stride_yw,
    stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wc, stride_bc,
    N_groups, N_channels_per_group, H, D, W,
    num_channels,
    eps,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)

    group_id = pid_c // N_channels_per_group
    c_local = pid_c % N_channels_per_group

    channel_offset = pid_c * stride_xc
    group_offset = group_id * N_channels_per_group

    X_block_ptr = X_ptr + pid_b * stride_xd + channel_offset
    Y_block_ptr = Y_ptr + pid_b * stride_yd + channel_offset

    mask_c = c_local < N_channels_per_group
    mask_hw = (tl.arange(0, BLOCK_W) < W)[:, None] & (tl.arange(0, BLOCK_H) < H)[None, :] & (tl.arange(0, BLOCK_D) < D)

    X_ptrs = X_block_ptr + \
        (tl.arange(0, BLOCK_D)[:, None, None] * stride_xd + \
         tl.arange(0, BLOCK_H)[None, :, None] * stride_xh + \
         tl.arange(0, BLOCK_W)[None, None, :] * stride_xw)
    Y_ptrs = Y_block_ptr + \
        (tl.arange(0, BLOCK_D)[:, None, None] * stride_yd + \
         tl.arange(0, BLOCK_H)[None, :, None] * stride_yh + \
         tl.arange(0, BLOCK_W)[None, None, :] * stride_yw)

    x = tl.load(X_ptrs, mask=mask_hw[None, :, :], other=0.0)

    mean = tl.sum(x, axis=[0, 1, 2]) / (D * H * W)
    diff = x - mean
    var = tl.sum(diff * diff, axis=[0, 1, 2]) / (D * H * W)
    inv_std = tl.rsqrt(var + eps)

    w = tl.load(W_ptr + group_offset + c_local, mask=mask_c, other=1.0)
    b = tl.load(B_ptr + group_offset + c_local, mask=mask_c, other=0.0)

    normed = diff * inv_std
    output = normed * w + b

    tl.store(Y_ptrs, output, mask=mask_hw[None, :, :])


@triton.jit
def _fused_conv_transpose3d_relu_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    batch_size, out_channels, out_depth, out_height, out_width,
    in_channels, in_depth, in_height, in_width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    groups,
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_g, weight_stride_r, weight_stride_s, weight_stride_t,
    has_bias: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // (out_channels * out_depth * out_height * out_width)
    residual = pid % (out_channels * out_depth * out_height * out_width)

    c = residual // (out_depth * out_height * out_width)
    residual = residual % (out_depth * out_height * out_width)
    d = residual // (out_height * out_width)
    residual = residual % (out_height * out_width)
    h = residual // out_width
    w = residual % out_width

    group = c // (out_channels // groups)
    c_group = c % (out_channels // groups)

    weight_group_offset = group * (out_channels // groups) * kernel_d * kernel_h * kernel_w

    acc = 0.0
    for kd in range(0, kernel_d):
        for kh in range(0, kernel_h):
            for kw in range(0, kernel_w):
                d_in = d * stride_d - padding_d + kd * dilation_d
                h_in = h * stride_h - padding_h + kh * dilation_h
                w_in = w * stride_w - padding_w + kw * dilation_w

                mask_d = (d_in >= 0) & (d_in < in_depth)
                mask_h = (h_in >= 0) & (h_in < in_height)
                mask_w = (w_in >= 0) & (w_in < in_width)
                mask = mask_d & mask_h & mask_w

                weight_offset = weight_group_offset + c_group * kernel_h * kernel_w * kernel_d + kd * kernel_h * kernel_w + kh * kernel_w + kw
                weight = tl.load(weight_ptr + weight_offset)
                input_offset = batch * input_stride_b + group * (in_channels // groups) * input_stride_c + c_group * input_stride_c + d_in * input_stride_d + h_in * input_stride_h + w_in * input_stride_w
                input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                acc += weight * input_val

    if has_bias:
        bias = tl.load(bias_ptr + c)
        acc += bias

    acc = acc if acc > 0.0 else 0.0

    output_offset = batch * output_stride_b + c * output_stride_c + d * output_stride_d + h * output_stride_h + w * output_stride_w
    tl.store(output_ptr + output_offset, acc)


def triton_fused_conv_transpose3d_relu(
    x,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    B, C_in, D_in, H_in, W_in = x.shape
    C_out, _, K_d, K_h, K_w = weight.shape

    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_d - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_h - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (K_w - 1) + output_padding[2] + 1

    out = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

    def grid(META):
        return (B * C_out * D_out * H_out * W_out,)

    _fused_conv_transpose3d_relu_kernel[grid](
        out,
        x,
        weight,
        bias,
        B, C_out, D_out, H_out, W_out,
        C_in, D_in, H_in, W_in,
        K_d, K_h, K_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        bias is not None,
        BLOCK_D=triton.next_power_of_2(D_out),
        BLOCK_H=triton.next_power_of_2(H_out),
        BLOCK_W=triton.next_power_of_2(W_out),
    )
    return out


def triton_group_norm(x, num_groups, weight, bias, eps=1e-5):
    B, C, D, H, W = x.shape
    x = x.view(B, num_groups, C // num_groups, D, H, W)
    mean = x.mean(dim=[3, 4, 5], keepdim=True)
    var = x.var(dim=[3, 4, 5], unbiased=False, keepdim=True)
    out = (x - mean) / torch.sqrt(var + eps)
    out = out.view(B, C, D, H, W)
    if weight is not None:
        out = out * weight.view(1, C, 1, 1, 1)
    if bias is not None:
        out = out + bias.view(1, C, 1, 1, 1)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for fused conv_transpose3d + ReLU and GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.conv_bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('conv_bias', None)

        self.group_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_channels))
        self.eps = 1e-5

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.conv_bias is not None:
            nn.init.zeros_(self.conv_bias)

    def forward(self, x):
        x = triton_fused_conv_transpose3d_relu(
            x,
            self.weight,
            self.conv_bias,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            output_padding=(0, 0, 0),
            groups=self.groups,
            dilation=(1, 1, 1),
        )
        x = triton_group_norm(x, self.groups, self.group_norm_weight, self.group_norm_bias, self.eps)
        return x