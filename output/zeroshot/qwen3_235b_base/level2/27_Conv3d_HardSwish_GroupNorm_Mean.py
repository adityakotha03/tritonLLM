import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Hardswish: x * relu6(x + 3) / 6
    x_plus_3 = x + 3.0
    zero = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)
    relu6 = tl.minimum(tl.maximum(x_plus_3, zero), 6.0)
    hardswish = x * relu6 / 6.0
    tl.store(out_ptr + offsets, hardswish, mask=mask)


def triton_hardswish(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def group_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    B,
    C,
    D,
    H,
    W,
    num_groups,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Each block handles one batch and one group
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    group_c = C // num_groups

    # Pointers to start of batch and group
    x_group_ptr = x_ptr + batch_idx * C * D * H * W + group_idx * group_c * D * H * W
    out_group_ptr = out_ptr + batch_idx * C * D * H * W + group_idx * group_c * D * H * W
    weight_group_ptr = weight_ptr + group_idx * group_c
    bias_group_ptr = bias_ptr + group_idx * group_c

    # Compute mean and variance over D, H, W for this group
    sum_val = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_HW), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_HW), dtype=tl.float32)
    N = D * H * W

    # Use blocks to tile over channels and spatial dimensions
    for ch_offset in range(0, group_c, BLOCK_SIZE_C):
        for hw_offset in range(0, N, BLOCK_SIZE_HW):
            ch_mask = (ch_offset + tl.arange(0, BLOCK_SIZE_C)) < group_c
            hw_mask = (hw_offset + tl.arange(0, BLOCK_SIZE_HW)) < N
            mask = ch_mask[:, None] & hw_mask[None, :]

            offsets = (ch_offset + tl.arange(0, BLOCK_SIZE_C))[:, None] * N + (hw_offset + tl.arange(0, BLOCK_SIZE_HW))[None, :]
            x_ptrs = x_group_ptr + offsets
            x = tl.load(x_ptrs, mask=mask, other=0.0)

            sum_val += tl.where(mask, x, 0.0)
            sum_sq += tl.where(mask, x * x, 0.0)

    # Reduce over spatial and channel dims
    mean = tl.sum(sum_val) / N / group_c
    var = tl.sum(sum_sq) / N / group_c - mean * mean

    # Normalize and apply affine transform
    for ch_offset in range(0, group_c, BLOCK_SIZE_C):
        for hw_offset in range(0, N, BLOCK_SIZE_HW):
            ch_mask = (ch_offset + tl.arange(0, BLOCK_SIZE_C)) < group_c
            hw_mask = (hw_offset + tl.arange(0, BLOCK_SIZE_HW)) < N
            mask = ch_mask[:, None] & hw_mask[None, :]

            offsets = (ch_offset + tl.arange(0, BLOCK_SIZE_C))[:, None] * N + (hw_offset + tl.arange(0, BLOCK_SIZE_HW))[None, :]
            x_ptrs = x_group_ptr + offsets
            x = tl.load(x_ptrs, mask=mask, other=0.0)

            x_norm = (x - mean) / tl.sqrt(var + eps)

            weight = tl.load(weight_group_ptr + tl.arange(0, BLOCK_SIZE_C), mask=ch_mask, other=1.0)
            bias = tl.load(bias_group_ptr + tl.arange(0, BLOCK_SIZE_C), mask=ch_mask, other=0.0)

            out = x_norm * weight[:, None] + bias[:, None]
            tl.store(out_group_ptr + offsets, out, mask=mask)


def triton_group_norm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    B, C, D, H, W = x.shape
    grid = (B, num_groups)
    BLOCK_SIZE_C = triton.next_power_of_2(group_c := C // num_groups)
    BLOCK_SIZE_HW = min(triton.next_power_of_2(H * W), 1024)
    group_norm_kernel[grid](
        x,
        weight,
        bias,
        None,  # running_mean
        None,  # running_var
        out,
        B,
        C,
        D,
        H,
        W,
        num_groups,
        eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return out


@triton.jit
def mean_pooling_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    D,
    H,
    W,
    BLOCK_SIZE_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    # Each block processes one batch
    x_batch_ptr = x_ptr + batch_idx * C * D * H * W
    out_batch_ptr = out_ptr + batch_idx * C

    N = D * H * W
    for ch_offset in range(0, C, BLOCK_SIZE_C):
        ch_mask = (ch_offset + tl.arange(0, BLOCK_SIZE_C)) < C
        offsets = (ch_offset + tl.arange(0, BLOCK_SIZE_C))[:, None] * N + tl.arange(0, N)[None, :]
        x_ptrs = x_batch_ptr + offsets
        mask = ch_mask[:, None]
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=1) / N
        tl.store(out_batch_ptr + ch_offset + tl.arange(0, BLOCK_SIZE_C), mean, mask=ch_mask)


def triton_mean_pooling(x):
    assert x.is_cuda
    x = x.contiguous()
    B, C, D, H, W = x.shape
    out = torch.empty((B, C), device=x.device, dtype=x.dtype)
    grid = (B,)
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    mean_pooling_kernel[grid](x, out, B, C, D, H, W, BLOCK_SIZE_C=BLOCK_SIZE_C)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for:
    - Hardswish activation
    - GroupNorm
    - Mean pooling over spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv(x)
        x = triton_hardswish(x)
        x = triton_group_norm(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.group_norm.eps)
        x = triton_mean_pooling(x)
        return x