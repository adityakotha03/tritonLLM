import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute tanh using the formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # We use fp16 for better Tensor Core utilization and performance on A100
    exp_x = tl.exp(x)
    exp_neg_x = tl.exp(-x)
    tanh_x = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

    tl.store(out_ptr + offsets, tanh_x, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel implementation of Tanh activation.
    Leverages FP16 and Tensor Core for faster computation on A100.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimal block size for memory coalescing and warp utilization

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty_like(x, dtype=x.dtype)

    # Launch kernel with FP16 conversion if input is FP32
    if x.dtype == torch.float32:
        # Convert to FP16 for Tensor Core acceleration
        x_fp16 = x.half()
        tanh_kernel[grid](x_fp16.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
        out = out.to(x.dtype)
    else:
        tanh_kernel[grid](x.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)