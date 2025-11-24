import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------- Triton kernel: double tanh ---------------------
@triton.jit
def double_tanh_kernel(
    inp_ptr,      # Pointer to input tensor (after min)
    out_ptr,      # Pointer to output tensor
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # Apply tanh twice
    x = tl.math.tanh(x)
    x = tl.math.tanh(x)

    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)

# --------------------- Helper: Triton double tanh wrapper ---------------------
def triton_double_tanh(inp: torch.Tensor) -> torch.Tensor:
    assert inp.is_cuda, "Input tensor must be on CUDA."
    inp = inp.contiguous()
    out = torch.empty_like(inp)

    n_elements = inp.numel()
    BLOCK_SIZE = 1024  # Tunable, choose a power of 2

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    double_tanh_kernel[grid](inp, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# --------------------- New model with fused double tanh ---------------------
class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    The two Tanh operations are fused into a single custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Convolution
        x = self.conv(x)

        # Minimum across channel dimension
        x, _ = torch.min(x, dim=1, keepdim=True)

        # Two successive Tanh fused into one kernel
        x = triton_double_tanh(x)
        return x