import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_clamp_kernel(
    x_ptr, out_ptr, scale, clamp_min, clamp_max,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x * scale
    x_clamped = tl.maximum(clamp_min, tl.minimum(clamp_max, x_scaled))
    tl.store(out_ptr + offsets, x_clamped, mask=mask)


def triton_mul_clamp(x: torch.Tensor, scale: float, clamp_min: float, clamp_max: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mul_clamp_kernel[grid](x, out, scale, clamp_min, clamp_max, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def maxpool_kernel_3d(
    x_ptr, out_ptr,
    in_depth, in_height, in_width,
    out_depth, out_height, out_width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    n_channels, batch_size,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_od = tl.program_id(2)

    offset_b = pid_b
    offset_c = pid_c

    batch_stride_x = in_depth * in_height * in_width
    channel_stride_x = in_height * in_width
    batch_stride_out = out_depth * out_height * out_width
    channel_stride_out = out_height * out_width

    for pid_oh in range(out_height // BLOCK_SIZE_HW):
        for pid_ow in range(out_width // BLOCK_SIZE_HW):
            acc = tl.full([BLOCK_SIZE_D, BLOCK_SIZE_HW, BLOCK_SIZE_HW], value=-float('inf'), dtype=tl.float32)

            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        x_d = pid_od * stride_d + kd
                        x_h_base = pid_oh * BLOCK_SIZE_HW * stride_h + kh
                        x_w_base = pid_ow * BLOCK_SIZE_HW * stride_w + kw

                        x_h = x_h_base + tl.arange(0, BLOCK_SIZE_HW)
                        x_w = x_w_base + tl.arange(0, BLOCK_SIZE_HW)

                        mask_hw = (x_h < in_height)[:, None] & (x_w < in_width)[None, :]
                        mask_d = (x_d < in_depth)[None, None]

                        mask = mask_d & mask_hw

                        offsets = (
                            offset_b * batch_stride_x +
                            offset_c * channel_stride_x +
                            x_d * in_height * in_width +
                            x_h[:, None] * in_width +
                            x_w[None, :]
                        )
                        x_vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

                        acc = tl.maximum(acc, x_vals)

            out_offsets = (
                offset_b * batch_stride_out +
                offset_c * channel_stride_out +
                pid_od * out_height * out_width +
                (pid_oh * BLOCK_SIZE_HW)[:, None] * out_width +
                (pid_ow * BLOCK_SIZE_HW)[None, :]
            )
            out_mask = ((tl.arange(0, BLOCK_SIZE_D) < out_depth)[:, None, None] &
                        (tl.arange(0, BLOCK_SIZE_HW) < out_height)[None, :, None] &
                        (tl.arange(0, BLOCK_SIZE_HW) < out_width)[None, None, :])
            tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_maxpool_3d(x: torch.Tensor, kernel_size, stride):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    kernel_d, kernel_h, kernel_w = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    stride_d, stride_h, stride_w = (stride,) if isinstance(stride, int) else stride

    in_b, in_c, in_d, in_h, in_w = x.shape
    out_d = (in_d - kernel_d) // stride_d + 1
    out_h = (in_h - kernel_h) // stride_h + 1
    out_w = (in_w - kernel_w) // stride_w + 1

    out = torch.empty((in_b, in_c, out_d, out_h, out_w), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_D = triton.next_power_of_2(out_d)
    BLOCK_SIZE_HW = min(32, max(16, triton.next_power_of_2(max(out_h, out_w))))

    grid = (in_b, in_c, out_d)
    maxpool_kernel_3d[grid](
        x_ptr=x,
        out_ptr=out,
        in_depth=in_d, in_height=in_h, in_width=in_w,
        out_depth=out_d, out_height=out_h, out_width=out_w,
        kernel_d=kernel_d, kernel_h=kernel_h, kernel_w=kernel_w,
        stride_d=stride_d, stride_h=stride_h, stride_w=stride_w,
        n_channels=in_c, batch_size=in_b,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for elementwise scale+clamp and custom 3D max pooling.
    The transposed convolution and global average pooling are kept as native PyTorch ops
    due to complexity and efficiency of existing implementations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_mul_clamp(x, self.scale, self.clamp_min, self.clamp_max)
        x = triton_maxpool_3d(x, self.maxpool_kernel_size, self.maxpool_kernel_size)
        x = self.global_avg_pool(x)
        return x