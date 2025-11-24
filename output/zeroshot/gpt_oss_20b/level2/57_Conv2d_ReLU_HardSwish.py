import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def hard_swish_kernel(
    out_ptr,
    in_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Applies ReLU followed by HardSwish:
        y = x * clamp((x + 3) / 6, 0, 1)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # ReLU
    x = tl.maximum(x, 0.0)

    # HardSwish
    hard_swish = x * tl.clip((x + 3.0) / 6.0, 0.0, 1.0)

    # Store result
    tl.store(out_ptr + offsets, hard_swish, mask=mask)


def triton_hard_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the fused ReLU + HardSwish Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    out = torch.empty_like(x)

    N = x.numel()
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hard_swish_kernel[grid](out, x, N, BLOCK_SIZE=256)
    return out


class ModelNew(nn.Module):
    """
    Optimized model: convolution followed by fused ReLU + HardSwish.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = triton_hard_swish(x)
        return x