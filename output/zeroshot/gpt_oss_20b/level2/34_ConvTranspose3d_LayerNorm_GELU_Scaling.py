import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------
# Triton kernel: GELU + scaling
# ------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["n"],
)
@triton.jit
def gelu_scale_kernel(
    x_ptr,          # Input pointer
    out_ptr,        # Output pointer
    scale,          # Scaling factor (scalar)
    n,              # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Approximate GELU: 0.5 * x * (1 + tanh[sqrt(2/pi)*(x + 0.044715*x^3)])
    sqrt2_over_pi = 0.7978845608028654
    inner = tl.math.tanh(sqrt2_over_pi * (x + 0.044715 * x * x * x))
    gelu = 0.5 * x * (1.0 + inner)

    out = gelu * scale
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_gelu_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Apply GELU activation followed by a scalar scaling using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n = x.numel()
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_scale_kernel[grid](x, out, scale, n, BLOCK_SIZE=256)
    return out


# ------------------------------------
# Optimized model
# ------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation,
    and scaling. The GELU+scaling step is fused into a custom Triton kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
        eps: float = 1e-5,
        scaling_factor: float = 1.0,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # 3D transposed convolution
        x = self.conv_transpose(x)

        # Layer normalization over channel dimension
        # Permute to (B, D, H, W, C) so LayerNorm normalizes last dim
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.layer_norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # GELU activation + scaling using Triton
        x = triton_gelu_scale(x, self.scaling_factor)

        return x