import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------
# Triton kernel: fused clamp + division
# --------------------------------------------
@triton.jit
def fused_clamp_div_kernel(
    inp_ptr,
    out_ptr,
    min_val,
    divisor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    # clamp
    inp = tl.max(inp, min_val)
    # divide
    inp = inp / divisor
    tl.store(out_ptr + offsets, inp, mask=mask)

def fused_clamp_div(inp: torch.Tensor, min_value: float, divisor: float):
    """
    Apply clamp(min_value) followed by division by divisor using a single Triton kernel.
    """
    assert inp.is_cuda, "Input must be on CUDA."
    out = torch.empty_like(inp)

    n_elements = inp.numel()
    BLOCK_SIZE = 1024  # Tunable; 1024 gives good occupancy on A100

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_clamp_div_kernel[grid](
        inp.contiguous(), out, min_value, divisor, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# --------------------------------------------
# Model with custom Triton kernel
# --------------------------------------------
class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value,
    and then divides the result by a constant. The clamp and division are fused into a single
    Triton kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True
        )
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        # 1. Transposed 3D convolution
        x = self.conv_transpose(x)

        # 2. Fused clamp + division using Triton
        x = fused_clamp_div(x, self.min_value, self.divisor)
        return x