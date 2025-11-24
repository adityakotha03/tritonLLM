import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------
# Triton kernel for fused LayerNorm
# ----------------------------------------------
@triton.jit
def layernorm_kernel(
    x_ptr,        # Base pointer to input tensor (batch, N)
    gamma_ptr,    # Base pointer to gamma   (N)
    beta_ptr,     # Base pointer to beta    (N)
    out_ptr,      # Base pointer to output tensor (batch, N)
    batch: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program instance per batch sample
    b = tl.program_id(0)
    base = b * N          # offset of the current sample in the flat view

    # ----- First pass: compute mean and variance -----------------
    i = 0
    sum_ = tl.zeros([1], dtype=tl.float32)
    sum_sq = tl.zeros([1], dtype=tl.float32)

    while i < N:
        offsets = base + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_ += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK_SIZE

    mean = sum_ / tl.float32(N)
    var  = sum_sq / tl.float32(N) - mean * mean
    std  = tl.sqrt(var + eps)

    # ----- Second pass: normalize and apply affine ---------------
    i = 0
    while i < N:
        offsets = base + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        g = tl.load(gamma_ptr + offsets - base, mask=mask, other=1.0)
        b_ = tl.load(beta_ptr + offsets - base, mask=mask, other=0.0)

        y = (x - mean) / std * g + b_
        tl.store(out_ptr + offsets, y, mask=mask)
        i += BLOCK_SIZE


# ----------------------------------------------
# Helper function that launches the Triton kernel
# ----------------------------------------------
def triton_layernorm(x: torch.Tensor,
                     gamma: torch.Tensor,
                     beta: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Compute LayerNorm on GPU using a fused Triton kernel.

    Parameters
    ----------
    x     : Tensor of shape (batch, *, * ... *)
    gamma : Tensor of shape (dim1, dim2, dim3, ...)  (same as x[1:])
    beta  : Tensor of shape (dim1, dim2, dim3, ...)  (same as x[1:])
    eps   : Small constant added to variance for numerical stability.

    Returns
    -------
    Tensor of same shape as ``x``.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda

    batch = x.shape[0]
    N = gamma.numel()           # number of elements per sample

    x_flat = x.reshape(batch, -1)
    gamma_flat = gamma.contiguous().view(-1)
    beta_flat = beta.contiguous().view(-1)

    out = torch.empty_like(x_flat)

    # One program per sample in the batch
    grid = lambda meta: (batch,)

    # Launch the kernel
    layernorm_kernel[grid](
        x_flat,
        gamma_flat,
        beta_flat,
        out,
        batch=batch,
        N=N,
        eps=eps,
        BLOCK_SIZE=256,
    )
    return out.reshape_as(x)


# ----------------------------------------------
# Optimized Model using the custom Triton kernel
# ----------------------------------------------
class ModelNew(nn.Module):
    """
    LayerNorm implemented with a fused Triton kernel.
    """

    def __init__(self, normalized_shape: tuple):
        """
        Parameters
        ----------
        normalized_shape : tuple
            The shape of the input tensor to be normalized.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        # Parameters for the affine transformation
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the custom Triton kernel.
        """
        return triton_layernorm(x, self.gamma, self.beta, eps=self.eps)