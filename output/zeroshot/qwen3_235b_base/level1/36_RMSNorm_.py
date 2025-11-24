import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,          # pointer to input tensor
    w_ptr,          # pointer to weight (not used here, kept for compatibility)
    out_ptr,        # pointer to output tensor
    batch_stride,   # stride for batch dimension
    feature_stride, # stride for feature dimension
    n_elements,     # number of elements per feature map (dim1 * dim2)
    num_features,   # number of features (size of dim 1)
    eps,            # epsilon value
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    batch_idx = tl.program_id(0)
    hw_idx = tl.program_id(1)

    # Compute offsets
    offset = batch_idx * batch_stride + hw_idx * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < (batch_idx * batch_stride + n_elements)

    # Load data for this feature slice
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Compute mean of squares
    mean_sq = tl.sum(x * x, axis=0) / num_features

    # Compute reciprocal of RMS (with epsilon)
    inv_rms = tl.math.rsqrt(mean_sq + eps)

    # Normalize and write back
    out = x * inv_rms
    tl.store(out_ptr + offset, out, mask=mask)

def triton_rms_norm(x: torch.Tensor, eps: float):
    assert x.is_cuda, "Input tensor must be on GPU."
    x = x.contiguous()

    batch_size, num_features, dim1, dim2 = x.shape
    n_elements = dim1 * dim2
    total_elements = x.numel()

    # Output tensor
    out = torch.empty_like(x)

    # Launch kernel with 2D grid: one dimension for batch, one for spatial (dim1 * dim2) blocks
    grid = lambda meta: (batch_size, triton.cdiv(n_elements, meta['BLOCK_SIZE']))

    # Heuristically choose block size
    BLOCK_SIZE = 1024
    while BLOCK_SIZE > n_elements and BLOCK_SIZE > 128:
        BLOCK_SIZE //= 2

    rms_norm_kernel[grid](
        x_ptr=x,
        w_ptr=None,
        out_ptr=out,
        batch_stride=num_features * n_elements,
        feature_stride=1,
        n_elements=n_elements,
        num_features=num_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized RMSNorm layer using Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton-optimized RMSNorm
        return triton_rms_norm(x, self.eps)