import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def div_kernel(
    x_ptr,
    out_ptr,
    divisor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x / divisor
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def add_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    bias_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Broadcast bias: bias has shape (C,1,1,1), x has (B,C,1,1,1) -> reshape bias
    bias_offsets = (offsets % bias_size) % bias_size  # cyclic indexing for broadcast
    bias = tl.load(bias_ptr + bias_offsets, mask=mask)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    reduce_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Reshape implicitly: sum over channel dim (dim=1), assume layout (B, C, 1, 1, 1)
    # Each block handles a slice; reduction over C
    # We reduce within the warp using tree reduction
    for s in range(1, BLOCK_SIZE):
        if s & (s - 1) == 0:  # power of 2
            x_next = tl.shuffle_down(x, s)
            if (offsets % reduce_size) + s < reduce_size:
                x = x + x_next
    # Only the first element in each reduction group is valid
    valid = (offsets % reduce_size) == 0
    tl.store(out_ptr + (offsets // reduce_size), x, mask=valid)


def triton_div(x: torch.Tensor, divisor: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    div_kernel[grid](x, out, divisor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_add_bias(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    bias_size = bias.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_bias_kernel[grid](x, bias, out, n_elements, bias_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_sum(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    # Assume dim=1 (channel), and input shape (B, C, 1, 1, 1)
    shape = x.shape
    out_shape = shape[:dim] + shape[dim+1:]
    out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    reduce_size = shape[dim]
    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sum_kernel[grid](x, out, n_elements, reduce_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for division, bias addition, and sum.
    Conv3d, MaxPool3d, and AdaptiveAvgPool3d are kept as PyTorch ops due to complexity
    and high optimization in cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = triton_div(x, self.divisor)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = triton_add_bias(x, self.bias)
        x = triton_sum(x, self.sum_dim)
        return x