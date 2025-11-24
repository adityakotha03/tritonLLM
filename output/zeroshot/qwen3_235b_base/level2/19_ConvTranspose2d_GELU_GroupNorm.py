import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh method
    x_sq = x * x
    x_cube = x_sq * x
    inner = 0.044715 * x_cube + x
    tanh_inner = tl.tanh(0.79788456 * inner)
    gelu_out = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + offsets, gelu_out, mask=mask)


def triton_gelu(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def group_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, HxW,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_c = C // num_groups

    offset_n = pid_n * C * HxW
    offset_g = pid_g * group_c * HxW

    c_range = tl.arange(0, BLOCK_SIZE_C)
    hw_range = tl.arange(0, BLOCK_SIZE_HW)

    mean = 0.0
    for c_idx in range(0, group_c, BLOCK_SIZE_C):
        for hw_idx in range(0, HxW, BLOCK_SIZE_HW):
            c_offsets = c_idx + c_range
            hw_offsets = hw_idx + hw_range
            mask = (c_offsets[:, None] < group_c) & (hw_offsets[None, :] < HxW)
            offsets = offset_n + offset_g + (c_offsets[:, None] * HxW + hw_offsets[None, :])
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            mean += tl.sum(tl.sum(x, axis=1), axis=0)

    mean = mean / (group_c * HxW)
    var = 0.0
    for c_idx in range(0, group_c, BLOCK_SIZE_C):
        for hw_idx in range(0, HxW, BLOCK_SIZE_HW):
            c_offsets = c_idx + c_range
            hw_offsets = hw_idx + hw_range
            mask = (c_offsets[:, None] < group_c) & (hw_offsets[None, :] < HxW)
            offsets = offset_n + offset_g + (c_offsets[:, None] * HxW + hw_offsets[None, :])
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            x_centered = x - mean
            var += tl.sum(tl.sum(x_centered * x_centered, axis=1), axis=0)

    inv_var = 1.0 / tl.sqrt(var / (group_c * HxW) + eps)

    for c_idx in range(0, group_c, BLOCK_SIZE_C):
        for hw_idx in range(0, HxW, BLOCK_SIZE_HW):
            c_offsets = c_idx + c_range
            hw_offsets = hw_idx + hw_range
            mask = (c_offsets[:, None] < group_c) & (hw_offsets[None, :] < HxW)
            offsets = offset_n + offset_g + (c_offsets[:, None] * HxW + hw_offsets[None, :])
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            x_norm = (x - mean) * inv_var

            weight = tl.load(weight_ptr + pid_g * group_c + c_offsets, mask=c_offsets < group_c, other=1.0)
            bias = tl.load(bias_ptr + pid_g * group_c + c_offsets, mask=c_offsets < group_c, other=0.0)

            output = x_norm * weight[:, None] + bias[:, None]
            tl.store(output_ptr + offsets, output, mask=mask)


class TritonGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)

        x = x.contiguous()
        N, C, H, W = x.shape
        HxW = H * W

        # Use small block sizes to fit in shared memory and avoid register pressure
        def grid(meta):
            return (N, self.num_groups)

        grid = lambda meta: (N, self.num_groups)

        output = torch.empty_like(x)
        group_norm_kernel[grid](
            x, self.weight, self.bias,
            output,
            N, C, HxW,
            self.num_groups,
            self.eps,
            BLOCK_SIZE_C=16,
            BLOCK_SIZE_HW=256,
        )
        return output


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for GELU and GroupNorm.
    ConvTranspose2d is kept as-is since it uses optimized cuDNN kernels.
    GELU and GroupNorm are replaced with Triton implementations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.group_norm = TritonGroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_gelu(x)
        x = self.group_norm(x)
        return x