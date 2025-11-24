import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,        # pointer to input tensor
    out_ptr,      # pointer to output tensor
    stride_x_b,   # stride for batch dimension
    stride_x_d,   # stride for the cumprod dimension
    numel_b,      # number of elements in batch dimension
    numel_d,      # number of elements in cumprod dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for batch
    batch_pid = tl.program_id(axis=0)
    # Offset for current batch
    x_batch_offset = batch_pid * stride_x_b
    out_batch_offset = batch_pid * stride_x_b

    # Load the first element along the cumprod dimension
    first_offset = x_batch_offset
    acc = tl.load(x_ptr + first_offset)

    # Store the first element
    tl.store(out_ptr + first_offset, acc)

    # Iterate over the rest of the elements in the cumprod dimension
    for i in range(1, numel_d):
        offset = x_batch_offset + i * stride_x_d
        x_val = tl.load(x_ptr + offset)
        acc = acc * x_val
        out_offset = out_batch_offset + i * stride_x_d
        tl.store(out_ptr + out_offset, acc)


def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton-based cumulative product along a given dimension.
    Currently supports dim=-1 or dim=1 for 2D tensors with dim=1 being the inner dimension.
    For generalization, we assume dim=1 for (B, D) shape.
    """
    assert x.dim() == 2, "Only 2D tensors supported for now"
    assert dim == 1, "Only cumprod along dim=1 is supported in this kernel"

    x = x.contiguous()
    out = torch.empty_like(x)

    batch_size, dim_size = x.shape

    # Use BLOCK_SIZE that is a power of 2 and covers typical sizes
    BLOCK_SIZE = triton.next_power_of_2(dim_size)

    # Each block handles one batch row
    grid = (batch_size,)

    cumprod_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        stride_x_b=x.stride(0),
        stride_x_d=x.stride(1),
        numel_b=batch_size,
        numel_d=dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for cumprod along dim=1.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim == 1 or self.dim == -1:
            return triton_cumprod(x, dim=self.dim)
        else:
            # Fall back to PyTorch for unsupported dims
            return torch.cumprod(x, dim=self.dim)