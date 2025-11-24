import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def tanh_sp_mul_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Softplus: log(1 + exp(x))
    # To avoid overflow, we use the numerically stable version:
    # softplus(x) = x if x > threshold else log(1 + exp(x))
    # We use threshold = 20 for stability
    threshold = 20.0
    x_clamped = tl.where(x > threshold, x, 0.0)
    exp_x = tl.exp(tl.where(x <= threshold, x, 0.0))
    log_term = tl.log(1.0 + exp_x)
    softplus = tl.where(x > threshold, x, log_term)

    # Tanh of softplus
    tanh_sp = tl.tanh(softplus)

    # Multiply by original x
    out = tanh_sp * x

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_tanh_sp_mul(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    tanh_sp_mul_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused activation using Triton.
    The operation: torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
    is replaced with a custom Triton kernel to reduce memory bandwidth usage and fuse operations.
    BatchNorm2d and Conv2d are kept as-is since they are already highly optimized in PyTorch,
    especially Conv2d which uses cuDNN and likely already leverages Tensor Cores.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = triton_tanh_sp_mul(x)
        x = self.bn(x)
        return x