import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------ Triton kernel for (x + add) * hard_swish(x + add) ------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def add_hardswish_kernel(
    x_ptr,
    add_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    add = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    y = x + add

    # hard_swish: y * relu6(y + 3) / 6
    relu6 = tl.max(0.0, tl.min(6.0, y + 3.0))
    y_hw = y * (relu6 / 6.0)

    tl.store(out_ptr + offsets, y_hw, mask=mask)

def add_hardswish(x: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of (x + add) * hard_swish(x + add).
    """
    assert x.is_cuda and add.is_cuda
    x = x.contiguous()
    add = add.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    add_hardswish_kernel[grid](x, add, out, n_elements, BLOCK_SIZE=256)
    return out

# ------------------ Optimized Model ------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor,
    and applies HardSwish activation using a fused Triton kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)                     # (b, out_c, D, H, W)
        x = add_hardswish(x, add_input)                # fused add + hard_swish
        return x