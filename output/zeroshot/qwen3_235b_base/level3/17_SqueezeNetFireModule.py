import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, height, width, in_channels, out_channels,
    input_height, input_width, output_height, output_width,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    input_stride_b, input_stride_h, input_stride_w, input_stride_c,
    weight_stride_oh, weight_stride_ow, weight_stride_ic, weight_stride_oc,
    output_stride_b, output_stride_h, output_stride_w, output_stride_c,
    is_1x1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(batch * output_height * output_width, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(in_channels, BLOCK_SIZE_K)

    group_size_m = GROUP_SIZE_M
    group_id = pid // group_size_m
    first_pid_m = group_id * group_size_m
    program_id_m = first_pid_m + (pid % group_size_m)
    program_id_n = pid // (num_programs_m)

    offs_m = program_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = program_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < out_channels
    mask_n = offs_n < batch * output_height * output_width

    offs_b = offs_n // (output_height * output_width)
    offs_y = (offs_n % (output_height * output_width)) // output_width
    offs_x = (offs_n % output_width)

    input_offs_b = offs_b * input_stride_b
    output_offs_bn = offs_n * output_stride_c

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        weight_offs_k = k_offs[:, None] * weight_stride_ic + offs_m[None, :] * weight_stride_oc
        mask_k = k_offs < in_channels
        mask_k = mask_k[:, None] & mask_m[None, :]

        weight = tl.load(weight_ptr + weight_offs_k, mask=mask_k, other=0.0)

        if is_1x1:
            input_offs_c = k_offs * input_stride_c
            input_offs_yx = offs_y * input_stride_h + offs_x * input_stride_w
            input_offs = input_offs_b[:, None] + input_offs_yx[:, None] + input_offs_c[None, :]
            mask_c = k_offs < in_channels
            mask_in = mask_c[None, :] & mask_n[:, None]
            input_vals = tl.load(input_ptr + input_offs, mask=mask_in, other=0.0)
            acc += tl.dot(weight, input_vals)
        else:
            for di in range(3):
                for dj in range(3):
                    p_y = offs_y * stride_h - padding_h + di * dilation_h
                    p_x = offs_x * stride_w - padding_w + dj * dilation_w
                    mask_y = (p_y >= 0) & (p_y < input_height)
                    mask_x = (p_x >= 0) & (p_x < input_width)
                    mask_p = mask_y & mask_x
                    input_offs_c = k_offs * input_stride_c
                    input_offs_yx = p_y * input_stride_h + p_x * input_stride_w
                    input_offs = input_offs_b[:, None] + input_offs_yx[:, None] + input_offs_c[None, :]
                    mask_c = k_offs < in_channels
                    mask_in = mask_c[None, :] & mask_n[:, None] & mask_p[:, None]
                    input_vals = tl.load(input_ptr + input_offs, mask=mask_in, other=0.0)
                    weight_offs_ij = di * weight_stride_oh + dj * weight_stride_ow
                    weight_slice = tl.load(weight_ptr + weight_offs_k + weight_offs_ij, mask=mask_k, other=0.0)
                    acc += tl.dot(weight_slice, input_vals)

    acc = acc.to(tl.float16)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
        acc += bias[:, None]

    out_relu = acc * (acc > 0)

    output_offs_mn = output_offs_bn[:, None] + offs_m[None, :] * output_stride_c
    mask_out = mask_m[None, :] & mask_n[:, None]
    tl.store(output_ptr + output_offs_mn, out_relu, mask=mask_out)


def fused_conv2d_relu(input, weight, bias, stride, padding, dilation, groups, is_1x1):
    assert groups == 1, "Grouped conv not supported"
    batch, in_channels, height, width = input.shape
    out_channels, _, kh, kw = weight.shape
    if kh == 1 and kw == 1:
        out_height = height
        out_width = width
    else:
        out_height = (height + 2 * padding[0] - kh) // stride[0] + 1
        out_width = (width + 2 * padding[1] - kw) // stride[1] + 1

    output = torch.empty((batch, out_channels, out_height, out_width), device=input.device, dtype=torch.float16)
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    def grid(META):
        return (triton.cdiv(out_channels, META['BLOCK_SIZE_M']) * triton.cdiv(batch * out_height * out_width, META['BLOCK_SIZE_N']),)

    fused_conv2d_relu_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else 0,
        output_ptr=output.data_ptr(),
        batch=batch,
        height=height,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=height,
        input_width=width,
        output_height=out_height,
        output_width=out_width,
        stride_h=stride[0],
        stride_w=stride[1],
        padding_h=padding[0],
        padding_w=padding[1],
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        input_stride_b=input.stride(0),
        input_stride_h=input.stride(2),
        input_stride_w=input.stride(3),
        input_stride_c=input.stride(1),
        weight_stride_oh=weight.stride(0),
        weight_stride_ow=weight.stride(1),
        weight_stride_ic=weight.stride(2),
        weight_stride_oc=weight.stride(3),
        output_stride_b=output.stride(0),
        output_stride_h=output.stride(2),
        output_stride_w=output.stride(3),
        output_stride_c=output.stride(1),
        is_1x1=is_1x1,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand1x1_channels = expand1x1_channels
        self.expand3x3_channels = expand3x3_channels

        self.squeeze_weight = nn.Parameter(torch.empty(squeeze_channels, in_channels, 1, 1))
        self.squeeze_bias = nn.Parameter(torch.empty(squeeze_channels))

        self.expand1x1_weight = nn.Parameter(torch.empty(expand1x1_channels, squeeze_channels, 1, 1))
        self.expand1x1_bias = nn.Parameter(torch.empty(expand1x1_channels))

        self.expand3x3_weight = nn.Parameter(torch.empty(expand3x3_channels, squeeze_channels, 3, 3))
        self.expand3x3_bias = nn.Parameter(torch.empty(expand3x3_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.squeeze_weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand1x1_weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand3x3_weight, nonlinearity='relu')
        nn.init.zeros_(self.squeeze_bias)
        nn.init.zeros_(self.expand1x1_bias)
        nn.init.zeros_(self.expand3x3_bias)

    def forward(self, x):
        x = fused_conv2d_relu(
            x, self.squeeze_weight, self.squeeze_bias,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, is_1x1=True
        )
        out1 = fused_conv2d_relu(
            x, self.expand1x1_weight, self.expand1x1_bias,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, is_1x1=True
        )
        out2 = fused_conv2d_relu(
            x, self.expand3x3_weight, self.expand3x3_bias,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, is_1x1=False
        )
        return torch.cat([out1, out2], dim=1)