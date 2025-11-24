import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def elu_kernel(
    x_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that applies the ELU activation function.

    Parameters
    ----------
    x_ptr : pointer
        Pointer to the input tensor.
    out_ptr : pointer
        Pointer to the output tensor.
    alpha : float
        ELU alpha value.
    n_elements : int
        Total number of elements in the tensors.
    BLOCK_SIZE : int (constexpr)
        Number of elements processed per program.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute ELU: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    out = tl.where(x > 0.0, x, alpha * (tl.math.exp(x) - 1.0))

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Apply ELU activation to a tensor using the custom Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must be CUDA and contiguous).
    alpha : float
        ELU alpha parameter.

    Returns
    -------
    torch.Tensor
        Output tensor with ELU applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    # Grid configuration for Triton
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    elu_kernel[grid](x, out, alpha, n_elements, BLOCK_SIZE=128)

    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces the PyTorch ELU with a custom Triton kernel.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu(x, self.alpha)