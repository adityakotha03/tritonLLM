import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------
# Element‑wise kernels
# -------------------------------------------------
@triton.jit
def subtract_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a - b, mask=mask)


@triton.jit
def hardswish_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    relu6 = tl.math.relu6(a + 3.0) / 6.0
    tl.store(out_ptr + offsets, a * relu6, mask=mask)


@triton.jit
def mish_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    tanh = tl.math.tanh(tl.math.log1p(tl.math.exp(a)))
    tl.store(out_ptr + offsets, a * tanh, mask=mask)


# -------------------------------------------------
# MaxPool kernel (2x2, stride 2)
# -------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global index
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)

    in_h = out_h * 2
    in_w = out_w * 2

    # strides
    stride_n = channels * height * width
    stride_c = height * width
    stride_h = width
    stride_w = 1

    base = batch_idx * stride_n + channel_idx * stride_c + in_h * stride_h + in_w * stride_w

    # Load the 2x2 patch
    vals = tl.load(input_ptr + base, mask=tl.arange(0, 4) < 4, other=-1e9)
    max_val = tl.maximum(vals[0], vals[1])
    max_val = tl.maximum(max_val, vals[2])
    max_val = tl.maximum(max_val, vals[3])

    out_offset = batch_idx * stride_n + channel_idx * stride_c + out_h * (width // 2) + out_w
    tl.store(output_ptr + out_offset, max_val)


# -------------------------------------------------
# Helper wrappers
# -------------------------------------------------
def triton_subtract(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    subtract_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_hardswish(a: torch.Tensor):
    assert a.is_cuda
    a = a.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_kernel[grid](a, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_mish(a: torch.Tensor):
    assert a.is_cuda
    a = a.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mish_kernel[grid](a, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_maxpool2d(x: torch.Tensor, kernel_size=2, stride=2):
    assert x.is_cuda
    n, c, h, w = x.shape
    out_h = h // stride
    out_w = w // stride
    out = torch.empty((n, c, out_h, out_w), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        n,
        c,
        out_h,
        out_w,
    )

    maxpool2d_kernel[grid](
        x,
        out,
        n,
        c,
        h,
        w,
        BLOCK_SIZE=128,
    )
    return out


# -------------------------------------------------
# Optimized model
# -------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for all element‑wise and pooling operations.
    The convolution uses the highly optimised Torch implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        subtract_value: float,
        pool_kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = triton_subtract(x, torch.full_like(x, self.subtract_value))
        x = triton_hardswish(x)
        x = triton_maxpool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
        x = triton_mish(x)
        return x