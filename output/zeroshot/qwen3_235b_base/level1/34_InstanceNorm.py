import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    mean_ptr, rstd_ptr,
    N, C, HxW,
    stride_xn, stride_xc,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_channels = C
    if pid >= num_channels * N:
        return

    # Determine batch and channel
    n = pid // num_channels
    c = pid % num_channels

    # Pointers to input/output for this (n, c) slice
    x = x_ptr + n.to(tl.int64) * stride_xn + c.to(tl.int64) * stride_xc
    y = y_ptr + n.to(tl.int64) * stride_xn + c.to(tl.int64) * stride_xc

    # Compute mean
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    mask = tl.arange(0, BLOCK_SIZE) < HxW
    for i in range(0, HxW, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        x_ptrs = x + offsets
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        mean += x_vals
    mean = tl.sum(mean) / HxW

    # Compute variance (for rstd)
    var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, HxW, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        x_ptrs = x + offsets
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_centered = x_vals - mean
        var += x_centered * x_centered
    var = tl.sum(var) / HxW
    rstd = 1.0 / tl.sqrt(var + eps)

    # Write mean and rstd if pointers are provided
    if mean_ptr is not None:
        mean_ptr += n * num_channels + c
        tl.store(mean_ptr, mean)
    if rstd_ptr is not None:
        rstd_ptr += n * num_channels + c
        tl.store(rstd_ptr, rstd)

    # Apply normalization and affine transform
    weight = tl.load(w_ptr + c) if w_ptr else 1.0
    bias = tl.load(b_ptr + c) if b_ptr else 0.0

    for i in range(0, HxW, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        x_ptrs = x + offsets
        y_ptrs = y + offsets
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_centered = x_vals - mean
        y_vals = x_centered * rstd * weight + bias
        tl.store(y_ptrs, y_vals, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of InstanceNorm2d using a custom Triton kernel.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA."
        x = x.contiguous()
        N, C, H, W = x.shape
        HxW = H * W
        out = torch.empty_like(x)

        # Use a block size that is a power of 2 and covers typical spatial sizes
        BLOCK_SIZE = 1024
        grid = lambda meta: ((N * C + 1) * 32,)

        instance_norm_kernel[grid](
            x_ptr=x,
            w_ptr=self.weight,
            b_ptr=self.bias,
            y_ptr=out,
            mean_ptr=None,
            rstd_ptr=None,
            N=N, C=C, HxW=HxW,
            stride_xn=x.stride(0),
            stride_xc=x.stride(1),
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out