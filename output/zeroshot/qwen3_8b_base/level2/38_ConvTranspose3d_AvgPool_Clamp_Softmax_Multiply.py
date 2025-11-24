import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def clamp_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    clamp_min,  # Minimum value
    clamp_max,  # Maximum value
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x < clamp_min, clamp_min, x)
    x = tl.where(x > clamp_max, clamp_max, x)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute softmax in a block
    exp_x = tl.exp(x - tl.max(x))
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_clamp(x: torch.Tensor, min_val, max_val):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    clamp_kernel[grid](x, out, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softmax_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = triton_softmax(x)
        x = x.view(b, c, d, h, w)
        x = x * self.scale
        return x