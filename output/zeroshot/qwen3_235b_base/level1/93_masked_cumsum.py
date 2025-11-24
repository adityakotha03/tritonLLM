import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    stride_x_dim,
    stride_out_dim,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_block_ptr = x_ptr + offsets
    mask_block_ptr = mask_ptr + offsets
    out_block_ptr = output_ptr + offsets

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(0, DIM):
        offset_dim = i * stride_x_dim
        x = tl.load(x_block_ptr + offset_dim, mask=mask, other=0.0)
        m = tl.load(mask_block_ptr + offset_dim, mask=mask, other=0.0)
        x_masked = x * m
        acc += x_masked
        tl.store(out_block_ptr + i * stride_out_dim, acc, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "x and mask must have the same shape"
    x = x.contiguous()
    mask = mask.contiguous()

    out = torch.empty_like(x)
    n_elements = 1
    for i in range(x.ndim):
        if i != dim:
            n_elements *= x.shape[i]

    stride_x_dim = x.stride(dim)
    stride_out_dim = out.stride(dim)
    DIM = x.shape[dim]

    # Choose block size as power of 2
    BLOCK_SIZE = triton.next_power_of_2(DIM)

    # Grid spans the non-reduction dimensions
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    masked_cumsum_kernel[grid](
        x_ptr=x,
        mask_ptr=mask,
        output_ptr=out,
        n_elements=n_elements,
        stride_x_dim=stride_x_dim,
        stride_out_dim=stride_out_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM=DIM,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        x = x.cuda()
        mask = mask.cuda()
        return triton_masked_cumsum(x, mask, self.dim)