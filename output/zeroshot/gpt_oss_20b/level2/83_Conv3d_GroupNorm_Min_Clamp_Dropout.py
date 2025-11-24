import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------ Triton kernel for clamp ------------------ #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def clamp_kernel(
    x_ptr,
    min_val,
    max_val,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    min_val_f = tl.full((BLOCK_SIZE,), min_val, dtype=tl.float32)
    max_val_f = tl.full((BLOCK_SIZE,), max_val, dtype=tl.float32)

    y = tl.minimum(x, min_val_f)
    y = tl.maximum(y, min_val_f)
    y = tl.minimum(y, max_val_f)

    tl.store(out_ptr + offsets, y, mask=mask)


def triton_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    N = x.numel()
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    clamp_kernel[grid](
        x, min_val, max_val, out, N, BLOCK_SIZE=256
    )
    return out


# ------------------ Optimized Model ------------------ #
class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, group normalization,
    clamping via a custom Triton kernel, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        min_value: float,
        max_value: float,
        dropout_p: float,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = triton_clamp(x, self.min_value, self.max_value)
        x = self.dropout(x)
        return x