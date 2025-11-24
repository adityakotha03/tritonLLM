import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    reduce_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x_block = x_ptr + offset
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(reduce_size):
        x = tl.load(x_block + i * n_elements, mask=mask, other=0.0)
        acc += x

    tl.store(out_ptr + offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."
        x = x.contiguous()
        shape = x.shape
        dim = self.dim

        # Handle negative indexing
        if dim < 0:
            dim = len(shape) + dim

        # Reshape to flatten dimensions before and after reduce_dim
        pre_dim = 1
        for i in range(dim):
            pre_dim *= shape[i]
        post_dim = 1
        for i in range(dim + 1, len(shape)):
            post_dim *= shape[i]
        reduce_size = shape[dim]

        x_reshaped = x.view(pre_dim, reduce_size, post_dim)
        x_transposed = x_reshaped.transpose(0, 1).contiguous()  # (reduce_size, pre_dim * post_dim)
        x_final = x_transposed.view(reduce_size, pre_dim * post_dim)

        n_elements = pre_dim * post_dim
        out = torch.empty((n_elements,), device=x.device, dtype=x.dtype)

        # Choose block size (power of 2, fits in register and shared memory limits)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        sum_kernel[grid](
            x_final,
            out,
            n_elements=n_elements,
            reduce_size=reduce_size,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Reshape output back to original shape with keepdim=True
        out_reshaped = out.view(pre_dim, post_dim).unsqueeze(dim)
        return out_reshaped.view(*[shape[i] if i != dim else 1 for i in range(len(shape))])