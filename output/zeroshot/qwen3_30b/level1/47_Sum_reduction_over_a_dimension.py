import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    batch_size,
    dim1,
    dim2,
    reduce_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Determine which batch and spatial indices we are processing
    batch_idx = pid // (dim1 if reduce_dim == 1 else dim2)
    spatial_idx = pid % (dim1 if reduce_dim == 1 else dim2)
    
    # Compute starting position for this block
    start_x = batch_idx * x_stride_0 + spatial_idx * x_stride_1
    start_out = batch_idx * out_stride_0 + spatial_idx * out_stride_1

    # Define offsets for the reduction dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (dim1 if reduce_dim == 1 else dim2)

    # Accumulate the sum over the reduction dimension
    acc = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    for i in range(0, (dim1 if reduce_dim == 1 else dim2), BLOCK_SIZE):
        # Calculate effective offset
        off = i + offsets
        valid = off < (dim1 if reduce_dim == 1 else dim2)
        # Load data with masking
        x = tl.load(x_ptr + start_x + off * (x_stride_2 if reduce_dim == 1 else x_stride_1), mask=valid, other=0.0)
        acc += x

    # Store the result
    tl.store(out_ptr + start_out, acc, mask=mask)


def triton_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Calculate output shape
    shape = list(x.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=x.dtype, device=x.device)

    # Determine strides
    x_stride_0 = x.stride(0)
    x_stride_1 = x.stride(1)
    x_stride_2 = x.stride(2)
    out_stride_0 = out.stride(0)
    out_stride_1 = out.stride(1)
    out_stride_2 = out.stride(2)

    # Number of elements to reduce
    reduce_size = x.shape[dim]
    total_blocks = (x.shape[0] * (x.shape[1] if dim == 2 else x.shape[2])) if dim == 1 else (x.shape[0] * x.shape[1])
    BLOCK_SIZE = 128  # Tunable parameter

    # Grid configuration
    grid = lambda meta: (total_blocks,)

    # Launch kernel
    sum_kernel[grid](
        x,
        out,
        x_stride_0,
        x_stride_1,
        x_stride_2,
        out_stride_0,
        out_stride_1,
        out_stride_2,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum(x, self.dim)