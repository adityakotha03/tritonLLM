import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_bias_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, out_channels, out_height, out_width,
    in_channels, in_height, in_width, kernel_size,
    stride, padding,
    in_stride_b, in_stride_c, in_stride_h, in_stride_w,
    weight_stride_kc, weight_stride_rc, weight_stride_rh, weight_stride_rw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(out_height * out_width, BLOCK_SIZE_N)
    num_programs_k = tl.cdiv(in_channels, BLOCK_SIZE_K)
    group_size_m = GROUP_SIZE_M
    group_id = pid // group_size_m
    first_pid_m = group_id * group_size_m
    group_size_m = min(num_programs_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = pid // (num_programs_m * group_size_m // group_size_m)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_bn = offs_n // (out_height * out_width)
    offs_hn = (offs_n % (out_height * out_width)) // out_width
    offs_wn = (offs_n % out_width)

    h_start = offs_hn * stride - padding
    w_start = offs_wn * stride - padding
    h_offset = h_start[:, None] + offs_k[None, :] // kernel_size
    w_offset = w_start[:, None] + offs_k[None, :] % kernel_size
    mask_hw = (h_offset >= 0) & (h_offset < in_height) & (w_offset >= 0) & (w_offset < in_width)

    h_offset = tl.where(mask_hw, h_offset, 0)
    w_offset = tl.where(mask_hw, w_offset, 0)

    bias_ptrs = bias_ptr + offs_m * tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    weight_ptrs = weight_ptr + (offs_m[:, None] * weight_stride_kc + offs_k[None, :] * weight_stride_rc +
                                (h_offset // kernel_size) * weight_stride_rh + (w_offset % kernel_size) * weight_stride_rw)
    x_ptrs = x_ptr + (offs_bn[:, None] * in_stride_b + offs_k[None, :] * in_stride_c +
                      h_offset * in_stride_h + w_offset * in_stride_w)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_channels, BLOCK_SIZE_K):
        mask_k = (offs_k[None, :] < in_channels - k)
        w = tl.load(weight_ptrs, mask=mask_k[None, :], other=0.0)
        x = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
        acc += tl.dot(w, x.to(tl.float32), out_dtype=tl.float32)
        weight_ptrs += BLOCK_SIZE_K * weight_stride_rc
        x_ptrs += BLOCK_SIZE_K * in_stride_c

    acc = acc.to(tl.float32)
    bias = tl.load(bias_ptrs, mask=offs_m < out_channels, other=0.0)
    acc += bias[:, None]
    acc = acc + 0  # no-op to materialize
    acc = tl.where(acc > 0, acc, 0.0)
    output_ptrs = output_ptr + (offs_bn[:, None] * out_stride_b + offs_m[:, None] * out_stride_c +
                                offs_hn[:, None] * out_stride_h + offs_wn[:, None] * out_stride_w)
    mask_output = (offs_m[:, None] < out_channels) & (offs_bn[:, None] < batch) & \
                  (offs_hn[:, None] < out_height) & (offs_wn[:, None] < out_width)
    tl.store(output_ptrs, acc, mask=mask_output)


def triton_conv_bias_relu(x, weight, bias, stride=1, padding=1):
    in_dtype = x.dtype
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    batch, _, in_height, in_width = x.shape
    out_height = (in_height + 2 * padding - kernel_h) // stride + 1
    out_width = (in_width + 2 * padding - kernel_w) // stride + 1

    output = torch.empty((batch, out_channels, out_height, out_width), dtype=in_dtype, device=x.device)

    def grid(META):
        return (triton.cdiv(out_channels, META['BLOCK_SIZE_M']) * triton.cdiv(out_height * out_width, META['BLOCK_SIZE_N']),)

    _conv_bias_relu_kernel[grid](
        x_ptr=x, weight_ptr=weight, bias_ptr=bias,
        output_ptr=output,
        batch=batch, out_channels=out_channels, out_height=out_height, out_width=out_width,
        in_channels=in_channels, in_height=in_height, in_width=in_width,
        kernel_size=kernel_h,
        stride=stride, padding=padding,
        in_stride_b=x.stride(0), in_stride_c=x.stride(1), in_stride_h=x.stride(2), in_stride_w=x.stride(3),
        weight_stride_kc=weight.stride(0), weight_stride_rc=weight.stride(1),
        weight_stride_rh=weight.stride(2), weight_stride_rw=weight.stride(3),
        out_stride_b=output.stride(0), out_stride_c=output.stride(1),
        out_stride_h=output.stride(2), out_stride_w=output.stride(3),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel for Conv2d + Bias + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.stride = 1
        self.padding = kernel_size // 2

    def forward(self, x):
        return triton_conv_bias_relu(x, self.weight, self.bias, stride=self.stride, padding=self.padding)