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
    # HardSwish: x * relu6(x + 3) / 6
    relu6 = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    out = x * relu6 / 6.0
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardswish(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that replaces the final HardSwish activation with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = self.group_norm(x)
        x = triton_hardswish(x)   # Triton-accelerated HardSwish
        return x