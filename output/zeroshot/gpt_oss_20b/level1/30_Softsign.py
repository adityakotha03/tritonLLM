import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernels for Softsign and its gradient
# --------------------------------------------------------------------------- #
BLOCK_SIZE = 1024  # Tune this value if desired


@triton.jit
def softsign_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element‑wise Softsign kernel.

    f(x) = x / (1 + |x|)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    denom = 1.0 + tl.abs(x)
    out = x / denom
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def softsign_grad_kernel(
    x_ptr,
    grad_out_ptr,
    grad_in_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gradient kernel for Softsign.

    grad_input = grad_output * 1 / (1 + |x|)^2
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    denom = 1.0 + tl.abs(x)
    denom_sq = denom * denom
    grad_in = grad_out / denom_sq
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


# --------------------------------------------------------------------------- #
# Custom autograd function that uses the above kernels
# --------------------------------------------------------------------------- #
class SoftsignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # Ensure the tensor is on GPU and contiguous
        assert input.is_cuda, "Softsign only supports CUDA tensors"
        input = input.contiguous()

        out = torch.empty_like(input)

        n = input.numel()
        grid = lambda meta: ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        softsign_kernel[grid](input, out, n, BLOCK_SIZE=BLOCK_SIZE)

        # Save for backward
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors

        grad_input = torch.empty_like(input)
        n = input.numel()
        grid = lambda meta: ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        grad_output = grad_output.contiguous()
        softsign_grad_kernel[grid](input, grad_output, grad_input, n, BLOCK_SIZE=BLOCK_SIZE)

        return grad_input


# Convenience wrapper
def triton_softsign(x: torch.Tensor) -> torch.Tensor:
    return SoftsignFunction.apply(x)


# --------------------------------------------------------------------------- #
# New model that uses the custom Triton implementation
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Simple model that performs a Softsign activation using a custom Triton kernel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softsign(x)