import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_plus_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, out_channels, out_height, out_width, in_channels, in_height, in_width,
    stride_h, stride_w, padding_h, padding_w, kernel_h, kernel_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_r, weight_stride_s,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(out_height * out_width, BLOCK_SIZE_N)
    num_programs_in_group = GROUP_SIZE_M * num_programs_n
    group_id = pid // num_programs_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_programs_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_programs_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_c = offs_k // (kernel_h * kernel_w)
    offs_k_r = (offs_k % (kernel_h * kernel_w)) // kernel_w
    offs_k_s = (offs_k % (kernel_h * kernel_w)) % kernel_w

    offs_output_hw = offs_n
    out_h = offs_output_hw // out_width
    out_w = offs_output_hw % out_width
    input_h = out_h * stride_h - padding_h + offs_k_r[:, None]
    input_w = out_w * stride_w - padding_w + offs_k_s[:, None]
    mask_input = (input_h >= 0) & (input_h < in_height) & (input_w >= 0) & (input_w < in_width)

    c_mask = offs_m[:, None] < out_channels
    hw_mask = offs_n < (out_height * out_width)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(in_channels, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k_c
        input_mask = (k_offs < in_channels)[:, None] & mask_input[None, :]
        input_ptrs = input_ptr + (
            (k_offs[:, None] * input_stride_c + input_h * input_stride_h + input_w * input_stride_w)
        )
        weight_ptrs = weight_ptr + (
            offs_m[:, None] * weight_stride_c + k_offs[None, :] * weight_stride_k +
            offs_k_r[:, None] * weight_stride_r + offs_k_s[:, None] * weight_stride_s
        )
        input_data = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=(k_offs[None, :] < in_channels)[:, :, None], other=0.0)
        accumulator += tl.dot(weight_data, input_data.to(tl.float16), out_dtype=tl.float32)

    acc = accumulator.to(tl.float16)
    bias_ptrs = bias_ptr + offs_m * 1
    bias = tl.load(bias_ptrs, mask=offs_m < out_channels, other=0.0).to(tl.float16)
    acc += bias[:, None]
    acc = acc + 0  # no activation for now (we don't fuse ReLU here to keep generality)

    output_ptrs = output_ptr + (
        offs_m[:, None] * output_stride_c + offs_output_hw[None, :] * output_stride_h
    )
    tl.store(output_ptrs, acc, mask=c_mask & hw_mask[None, :])


def triton_fused_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int = 1,
    groups: int = 1
):
    assert groups == 1, "Grouped conv not supported"
    assert dilation == 1, "Dilation not supported"
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_height = (in_height + 2 * padding - kernel_h) // stride + 1
    out_width = (in_width + 2 * padding - kernel_w) // stride + 1

    out = torch.empty((batch, out_channels, out_height, out_width), device=x.device, dtype=torch.float16)

    def grid(META):
        return (triton.cdiv(out_channels, META['BLOCK_SIZE_M']) * triton.cdiv(out_height * out_width, META['BLOCK_SIZE_N']),)

    fused_conv2d_plus_relu_kernel[grid](
        x, weight, bias, out,
        batch, out_channels, out_height, out_width, in_channels, in_height, in_width,
        stride, stride, padding, padding, kernel_h, kernel_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return out


@triton.jit
def channel_concat_kernel(
    x1_ptr, x2_ptr, x3_ptr, x4_ptr,
    out_ptr,
    n_elements_1, n_elements_2, n_elements_3, n_elements_4,
    offset_1, offset_2, offset_3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements_1 + n_elements_2 + n_elements_3 + n_elements_4)

    total_offset = 0
    if offsets < n_elements_1:
        data = tl.load(x1_ptr + offsets, mask=offsets < n_elements_1, other=0.0)
    else:
        offsets -= n_elements_1
        total_offset += n_elements_1
        if offsets < n_elements_2:
            data = tl.load(x2_ptr + offsets, mask=offsets < n_elements_2, other=0.0)
        else:
            offsets -= n_elements_2
            total_offset += n_elements_2
            if offsets < n_elements_3:
                data = tl.load(x3_ptr + offsets, mask=offsets < n_elements_3, other=0.0)
            else:
                offsets -= n_elements_3
                data = tl.load(x4_ptr + offsets, mask=offsets < n_elements_4, other=0.0)
    tl.store(out_ptr + offsets, data, mask=mask)


def triton_channel_concat(tensors):
    assert all(t.is_cuda for t in tensors)
    shapes = [t.shape for t in tensors]
    batch, _, height, width = shapes[0]
    assert all(s == (batch, -1, height, width) for s in shapes), "All tensors must have same batch, height, width"
    out_channels = sum(s[1] for s in shapes)
    out = torch.empty((batch, out_channels, height, width), device=tensors[0].device, dtype=tensors[0].dtype)

    n_elements = [t.numel() for t in tensors]
    offsets = [0, n_elements[0], n_elements[0] + n_elements[1], n_elements[0] + n_elements[1] + n_elements[2]]

    def grid(meta): return ((sum(n_elements) + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    channel_concat_kernel[grid](
        tensors[0].contiguous(), tensors[1].contiguous(), tensors[2].contiguous(), tensors[3].contiguous(),
        out,
        n_elements[0], n_elements[1], n_elements[2], n_elements[3],
        offsets[0], offsets[1], offsets[2],
        BLOCK_SIZE=1024
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1_weight = nn.Parameter(torch.empty(out_1x1, in_channels, 1, 1))
        self.branch1x1_bias = nn.Parameter(torch.zeros(out_1x1))
        
        # 3x3 convolution branch
        self.branch3x3_reduce_weight = nn.Parameter(torch.empty(reduce_3x3, in_channels, 1, 1))
        self.branch3x3_reduce_bias = nn.Parameter(torch.zeros(reduce_3x3))
        self.branch3x3_conv_weight = nn.Parameter(torch.empty(out_3x3, reduce_3x3, 3, 3))
        self.branch3x3_conv_bias = nn.Parameter(torch.zeros(out_3x3))
        
        # 5x5 convolution branch
        self.branch5x5_reduce_weight = nn.Parameter(torch.empty(reduce_5x5, in_channels, 1, 1))
        self.branch5x5_reduce_bias = nn.Parameter(torch.zeros(reduce_5x5))
        self.branch5x5_conv_weight = nn.Parameter(torch.empty(out_5x5, reduce_5x5, 5, 5))
        self.branch5x5_conv_bias = nn.Parameter(torch.zeros(out_5x5))
        
        # Max pooling branch
        self.pool_proj_weight = nn.Parameter(torch.empty(pool_proj, in_channels, 1, 1))
        self.pool_proj_bias = nn.Parameter(torch.zeros(pool_proj))

        self.in_channels = in_channels
        self.out_1x1 = out_1x1
        self.reduce_3x3 = reduce_3x3
        self.out_3x3 = out_3x3
        self.reduce_5x5 = reduce_5x5
        self.out_5x5 = out_5x5
        self.pool_proj = pool_proj

        # Initialize weights
        nn.init.kaiming_uniform_(self.branch1x1_weight)
        nn.init.kaiming_uniform_(self.branch3x3_reduce_weight)
        nn.init.kaiming_uniform_(self.branch3x3_conv_weight)
        nn.init.kaiming_uniform_(self.branch5x5_reduce_weight)
        nn.init.kaiming_uniform_(self.branch5x5_conv_weight)
        nn.init.kaiming_uniform_(self.pool_proj_weight)

    def forward(self, x):
        x = x.to(torch.float16)

        # Branch 1: 1x1 conv
        branch1x1 = triton_fused_conv2d(x, self.branch1x1_weight, self.branch1x1_bias, stride=1, padding=0)

        # Branch 2: 1x1 + 3x3
        x3 = triton_fused_conv2d(x, self.branch3x3_reduce_weight, self.branch3x3_reduce_bias, stride=1, padding=0)
        branch3x3 = triton_fused_conv2d(x3, self.branch3x3_conv_weight, self.branch3x3_conv_bias, stride=1, padding=1)

        # Branch 3: 1x1 + 5x5
        x5 = triton_fused_conv2d(x, self.branch5x5_reduce_weight, self.branch5x5_reduce_bias, stride=1, padding=0)
        branch5x5 = triton_fused_conv2d(x5, self.branch5x5_conv_weight, self.branch5x5_conv_bias, stride=1, padding=2)

        # Branch 4: MaxPool + 1x1
        pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = triton_fused_conv2d(pool, self.pool_proj_weight, self.pool_proj_bias, stride=1, padding=0)

        # Concatenate along channel dimension
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return triton_channel_concat(outputs)