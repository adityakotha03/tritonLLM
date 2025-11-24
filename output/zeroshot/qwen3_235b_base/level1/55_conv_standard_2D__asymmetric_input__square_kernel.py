import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, out_channels, out_height, out_width,
    in_channels, height, width,
    kernel_size,
    stride, padding, dilation,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s,
    output_stride_b, output_stride_k, output_stride_h, output_stride_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(out_width, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(in_channels // groups, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_k = tl.arange(0, BLOCK_SIZE_N)

    x_ptrs = x_ptr + (
        (offs_m[:, None] // groups) // (in_channels // groups) * input_stride_b +
        (offs_m[:, None] // groups) % (in_channels // groups) * input_stride_c
    )
    w_ptrs = w_ptr + (
        offs_m[:, None] * weight_stride_k +
        offs_n[None, :] * weight_stride_c
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for h in range(kernel_size):
        for w in range(kernel_size):
            h_pad = h * dilation - padding
            w_pad = w * dilation - padding
            h_base = h_pad + offs_k * stride
            w_base = w_pad + offs_k * stride
            mask_x = (
                (h_base >= 0) & (h_base < height) &
                (w_base >= 0) & (w_base < width)
            )
            offsets_x = (
                h_base[None, :] * input_stride_h +
                w_base[None, :] * input_stride_w
            )
            x_ptrs_hw = x_ptrs + offsets_x
            x = tl.load(x_ptrs_hw, mask=mask_x[None, :], other=0.0)
            w = tl.load(w_ptrs + h * weight_stride_r + w * weight_stride_s)
            accumulator += tl.dot(x, w)
    
    c = accumulator.to(tl.float16)

    offs_m_out = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_out = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    out_ptrs = out_ptr + (
        offs_m_out[:, None] * output_stride_k +
        offs_n_out[None, :] * output_stride_b
    )
    out_mask = (offs_m_out[:, None] < out_channels) & (offs_n_out[None, :] < batch_size * out_height * out_width)
    tl.store(out_ptrs, c, mask=out_mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  stride: int, padding: int, dilation: int, groups: int):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    assert kernel_size == weight.shape[2] == weight.shape[3], "Only square kernels supported"
    
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=torch.float16)
    x = x.to(torch.float16)
    weight = weight.to(torch.float16)

    def grid(META):
        return (triton.cdiv(out_channels, META['BLOCK_SIZE_M']) * triton.cdiv(in_channels // groups, META['BLOCK_SIZE_K']),)

    input_stride_b = x.stride(0)
    input_stride_c = x.stride(1)
    input_stride_h = x.stride(2)
    input_stride_w = x.stride(3)

    weight_stride_k = weight.stride(0)
    weight_stride_c = weight.stride(1)
    weight_stride_r = weight.stride(2)
    weight_stride_s = weight.stride(3)

    output_stride_b = out.stride(0)
    output_stride_k = out.stride(1)
    output_stride_h = out.stride(2)
    output_stride_w = out.stride(3)

    conv2d_kernel[grid](
        x, weight, out,
        batch_size, out_channels, out_height, out_width,
        in_channels, height, width,
        kernel_size,
        stride, padding, dilation,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s,
        output_stride_b, output_stride_k, output_stride_h, output_stride_w,
        groups,
        BLOCK_SIZE_M=64, BLOCK_SIZE_K=64, BLOCK_SIZE_N=32,
        GROUP_SIZE_M=8,
    )

    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias,
                             self.stride, self.padding, self.dilation, self.groups)