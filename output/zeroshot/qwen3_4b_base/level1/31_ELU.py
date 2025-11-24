import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def elu_kernel(
    x_ptr,
    alpha_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    alpha = tl.load(alpha_ptr, mask=mask, other=1.0)  # alpha is scalar, so load once

    # Compute ELU: x if x >= 0, else alpha * (exp(alpha * x) - 1)
    zero = tl.zeros_like(x)
    positive = x >= zero
    exp_term = tl.exp(alpha * x)
    elu = tl.where(positive, x, alpha * (exp_term - 1))

    tl.store(out_ptr + offsets, elu, mask=mask)


def triton_elu(x: torch.Tensor, alpha: float = 1.0):
    """
    Custom ELU kernel using Triton for better performance on A100.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Use FP16 for better Tensor Core utilization and speed
    # Convert to FP16 for computation, but keep input/output in original dtype
    x_fp16 = x.half() if x.dtype == torch.float32 else x
    alpha_fp16 = torch.tensor(alpha, dtype=torch.float16).cuda()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Power of 2, optimized for Ampere and shared memory

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    elu_kernel[grid](x_fp16.data_ptr(), alpha_fp16.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Convert output back to original dtype
    if x.dtype == torch.float32:
        out = out.float()
    elif x.dtype == torch.bfloat16:
        out = out.bfloat16()
    return out


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu(x, alpha=self.alpha)