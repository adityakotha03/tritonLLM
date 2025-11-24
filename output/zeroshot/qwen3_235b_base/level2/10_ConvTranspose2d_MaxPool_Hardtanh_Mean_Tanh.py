import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_tanh(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    input_height,
    input_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    num_hw_blocks = (input_height * input_width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW

    for block_hw in range(num_hw_blocks):
        offset_hw = block_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
        mask_hw = offset_hw < input_height * input_width
        offset_x = pid_c * input_height * input_width + offset_hw
        x = tl.load(x_ptr + offset_x, mask=mask_hw, other=0.0)
        mean_val = tl.sum(x, axis=0) / (input_height * input_width)
        tl.store(out_ptr + pid_c, mean_val)


def triton_mean_2d(x, dim):
    assert x.is_cuda
    x = x.contiguous()
    n, c, h, w = x.shape
    out = torch.empty(n, c, 1, 1, dtype=x.dtype, device=x.device)
    grid = (c,)
    mean_kernel[grid](
        x,
        out,
        n_channels=c,
        input_height=h,
        input_width=w,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_HW=256
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for mean and tanh operations.
    ConvTranspose2d and MaxPool2d are kept as native PyTorch ops due to complexity and high optimization already present.
    Hardtanh is also kept as native op since it's already efficient and not a bottleneck.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = triton_mean_2d(x, dim=(2, 3))
        x = triton_tanh(x)
        return x