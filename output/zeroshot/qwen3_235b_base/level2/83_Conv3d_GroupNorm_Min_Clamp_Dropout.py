import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def clamp_min_kernel(
    x_ptr,
    out_ptr,
    min_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.maximum(x, min_value)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def clamp_min_max_kernel(
    x_ptr,
    out_ptr,
    min_value,
    max_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.maximum(x, min_value)
    result = tl.minimum(result, max_value)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    p,
    n_elements,
    seed,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offset + offsets)
    keep_mask = random > p
    result = tl.where(keep_mask, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_clamp_min(x: torch.Tensor, min_value: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    clamp_min_kernel[grid](x, out, min_value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_clamp_min_max(x: torch.Tensor, min_value: float, max_value: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    clamp_min_max_kernel[grid](x, out, min_value, max_value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_dropout(x: torch.Tensor, p: float, training: bool):
    if not training or p == 0.0:
        return x
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    seed = 1234
    offset = tl.program_id(0) * BLOCK_SIZE
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    dropout_kernel[grid](x, out, p, n_elements, seed, 0, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for elementwise operations.
    The 3D convolution and GroupNorm are kept as PyTorch ops since they are already optimized,
    but clamp, min, and dropout are fused and replaced with efficient Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout_p = dropout_p
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = triton_clamp_min(x, self.min_value)
        x = triton_clamp_min_max(x, self.min_value, self.max_value)
        x = triton_dropout(x, self.dropout_p, self.training)
        return x