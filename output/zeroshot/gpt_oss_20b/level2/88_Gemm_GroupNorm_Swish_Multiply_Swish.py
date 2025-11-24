import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernel that fuses the two Swish activations and the
# element‑wise multiplication by the learnable weight.
# --------------------------------------------------------------
@triton.jit
def _fused_ops_kernel(
    x_ptr,          # input after GroupNorm (batch, out_features)
    weight_ptr,     # learnable weight (out_features,)
    out_ptr,        # output (batch, out_features)
    n_elements,     # total number of elements = batch * out_features
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the start index for this program
    start = tl.program_id(0) * BLOCK_SIZE
    # Index range for this block
    offsets = start + tl.arange(0, BLOCK_SIZE)
    # Mask for bounds
    mask = offsets < n_elements

    # Load input, weight and perform the first Swish
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets % (n_elements // mask.sum()), mask=mask, other=0.0)
    # Swish: x * sigmoid(x)
    swish1 = x * tl.sigmoid(x)

    # Multiply by weight
    mul = swish1 * w

    # Second Swish
    out = mul * tl.sigmoid(mul)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_ops(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around the Triton fused kernel.  Expects that `x` and `weight`
    are on the same CUDA device and are contiguous.
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _fused_ops_kernel[grid](
        x, weight, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# --------------------------------------------------------------
# The optimized model.
# --------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that replaces the two Swish + element‑wise
    multiplications with a single Triton kernel.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=True)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        # The learnable weight used in the element‑wise multiplication
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, in_features) -> (batch, out_features)
        x = self.gemm(x)
        # (batch, out_features) -> (batch, out_features)
        x = self.group_norm(x)
        # Fuse the following ops into one Triton kernel:
        #   x = x * sigmoid(x)
        #   x = x * self.multiply_weight
        #   x = x * sigmoid(x)
        x = triton_fused_ops(x, self.multiply_weight)
        return x