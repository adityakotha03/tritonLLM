import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel for GELU using tanh approximation
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    x2 = x * x
    x3 = x2 * x
    c = 0.044715
    inner = tl.math.sqrt(2.0 / tl.math.pi) * (x + c * x3)
    tanh_inner = tl.math.tanh(inner)
    gelu = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + offsets, gelu, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply GELU activation using the Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128  # tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU (Triton),
    and normalizes with GroupNorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_gelu(x)
        x = self.group_norm(x)
        return x