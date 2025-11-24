import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernels ----------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def logsumexp_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    n_samples,
    n_channels,
    n_height,
    n_width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes torch.logsumexp over the channel dimension (dim=1).
    Input and output are contiguous tensors with shape [B, C, H, W] and [B, 1, H, W] respectively.
    """
    # Each program handles one (b, h, w) position
    pos = tl.program_id(0)
    n_hw = n_height * n_width
    b = pos // n_hw
    hw = pos % n_hw
    h = hw // n_width
    w = hw % n_width

    # Compute base offset for this (b, h, w) position
    base_offset = b * stride_b + h * stride_h + w * stride_w

    # First pass: find maximum over channels
    max_val = tl.full([BLOCK_SIZE], -float("inf"))
    for i in range(0, n_channels, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_channels
        idx = base_offset + offsets * stride_c
        vals = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, vals)

    max_val = tl.max(max_val)

    # Second pass: compute sum of exp(x - max)
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, n_channels, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_channels
        idx = base_offset + offsets * stride_c
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_exp = sum_exp + tl.exp(vals - max_val)

    sum_exp = tl.sum(sum_exp)
    out_val = tl.log(sum_exp) + max_val

    # Store result
    out_offset = b * stride_b + h * stride_h + w * stride_w
    tl.store(out_ptr + out_offset, out_val)


# ---------- Helper wrappers ----------

def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ( (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=128)
    return out


def triton_logsumexp(x: torch.Tensor):
    """
    x: tensor of shape [B, C, H, W], contiguous
    returns: tensor of shape [B, 1, H, W]
    """
    assert x.is_cuda
    x = x.contiguous()
    B, C, H, W = x.shape
    out = torch.empty((B, 1, H, W), dtype=x.dtype, device=x.device)
    stride_b, stride_c, stride_h, stride_w = x.stride()
    n_samples = B * H * W
    grid = lambda meta: ( (n_samples + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    logsumexp_kernel[grid](
        x,
        out,
        stride_b,
        stride_c,
        stride_h,
        stride_w,
        n_samples,
        C,
        H,
        W,
        BLOCK_SIZE=128,
    )
    return out


# ---------- Optimized Model ----------

class ModelNew(nn.Module):
    """
    Optimized model with Triton kernels for residual addition and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        x_tanh = self.tanh(x_norm)
        x_hard_swish = self.hard_swish(x_tanh)
        x_res = triton_add(x_conv, x_hard_swish)
        x_logsumexp = triton_logsumexp(x_res)
        return x_logsumexp