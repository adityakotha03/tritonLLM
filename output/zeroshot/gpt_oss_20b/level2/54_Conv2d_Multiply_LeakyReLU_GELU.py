import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Fusion kernel: (x * multiplier) -> LeakyReLU -> GELU
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_act_kernel(
    x_ptr,
    multiplier_ptr,
    out_ptr,
    N, C, H, W,
    multiplier_shape,   # (C, 1, 1)
    BLOCK_SIZE: tl.constexpr,
    alpha: tl.constexpr,
):
    """
    Each program processes a contiguous block of elements across the entire
    (N, C, H, W) tensor.  The multiplier is broadcasted over H and W.
    """
    # Global index of the element this program will process
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    size = N * C * H * W
    mask = idx < size

    # Load input
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # Compute channel index to index into multiplier
    # idx = n*(C*H*W) + c*(H*W) + h*W + w
    c = (idx // (H * W)) % C
    mul_idx = c * multiplier_shape[1] * multiplier_shape[2] + 0  # multiplier is (C,1,1)

    mul = tl.load(multiplier_ptr + mul_idx, mask=mask, other=1.0)

    # Apply multiplier
    x = x * mul

    # LeakyReLU
    x = tl.where(x > 0, x, x * alpha)

    # GELU (approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
    sqrt_2_over_pi = 0.7978845608028654
    cst = 0.044715
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + cst * x_cubed)
    gelu = 0.5 * x * (1 + tl.tanh(tanh_arg))

    # Store result
    tl.store(out_ptr + idx, gelu, mask=mask)


def fused_act(x: torch.Tensor, multiplier: torch.Tensor, alpha: float = 0.01):
    """
    Wrapper for the fused activation kernel.
    """
    assert x.is_cuda and multiplier.is_cuda
    x = x.contiguous()
    multiplier = multiplier.contiguous()

    N, C, H, W = x.shape
    out = torch.empty_like(x)

    # grid size: number of blocks needed
    grid = lambda meta: ((N * C * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # launch kernel
    fused_act_kernel[grid](
        x,
        multiplier,
        out,
        N,
        C,
        H,
        W,
        multiplier.shape,
        alpha=alpha,
    )
    return out


# ----------------------------------------------------------------------
# New model definition
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar,
    applies LeakyReLU, and then GELU using a fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()  # kept for reference; not used in forward

    def forward(self, x):
        x = self.conv(x)
        x = fused_act(x, self.multiplier, alpha=self.leaky_relu.negative_slope)
        return x