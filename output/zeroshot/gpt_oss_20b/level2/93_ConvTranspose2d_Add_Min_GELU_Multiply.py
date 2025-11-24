import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel that fuses:  x = x + add_value
#                           x = min(x, 0)
#                           x = gelu(x)
#                           x = x * multiply_value
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def fused_postprocess_kernel(
    in_ptr,          # Pointer to the conv_transpose output
    out_ptr,         # Pointer to the output tensor
    add_value,       # Scalar add value
    multiply_value, # Scalar multiply value
    n,               # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Index of the current program (block)
    pid = tl.program_id(0)
    # Compute the start offset for this block
    start = pid * BLOCK_SIZE
    # Generate offsets for this block
    offsets = start + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out‑of‑bounds access
    mask = offsets < n
    # Load, compute, and store in one go
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Add
    x = x + add_value
    # Min with 0
    x = tl.min(x, 0.0)
    # GELU
    x = x * 0.5 * (1.0 + tl.math.erf(x / tl.math.sqrt(2.0)))
    # Multiply
    x = x * multiply_value
    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_postprocess(x: torch.Tensor, add_value: float, multiply_value: float) -> torch.Tensor:
    """
    Wrapper for the Triton kernel. Assumes `x` is contiguous and on CUDA.
    """
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    # Total number of elements in the tensor
    n = x.numel()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_postprocess_kernel[grid](
        x, out, float(add_value), float(multiply_value), n, BLOCK_SIZE=256
    )
    return out


# ----------------------------------------------------------------------
# Model that uses the custom Triton kernel for the post‑processing stage
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution followed by
    fused post‑processing (add, min, GELU, multiply) implemented with a
    Triton kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Transposed convolution
        x = self.conv_transpose(x)
        # Fused post‑processing
        x = fused_postprocess(x, self.add_value, self.multiply_value)
        return x