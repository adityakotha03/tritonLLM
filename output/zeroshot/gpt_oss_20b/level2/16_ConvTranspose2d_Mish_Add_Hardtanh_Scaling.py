import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_activation_kernel(
    x_ptr,            # input tensor pointer
    out_ptr,          # output tensor pointer
    n_elements,       # total number of elements
    add_value,        # scalar to add
    scale,            # scalar to multiply
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses Mish -> add -> Hardtanh -> scale.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # For numerical stability we clip exp argument
    exp_x = tl.exp(tl.min(x, 20.0))
    softplus = tl.log1p(exp_x)
    mish = x * tl.tanh(softplus)

    # Add
    mish = mish + add_value

    # Hardtanh
    mish = tl.maximum(tl.minimum(mish, 1.0), -1.0)

    # Scale
    mish = mish * scale

    # Store
    tl.store(out_ptr + offsets, mish, mask=mask)


def fused_activation(x: torch.Tensor, add_value: float, scale: float):
    """
    Apply Mish, add scalar, Hardtanh, and scaling in a single Triton kernel.
    """
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _fused_activation_kernel[grid](x, out, n_elements, add_value, scale, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    """
    Optimized model: ConvTranspose2d remains PyTorch's fast implementation,
    subsequent element‑wise ops are fused into one Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_activation(x, self.add_value, self.scale)
        return x