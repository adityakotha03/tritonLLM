import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that performs a min reduction over one dimension of a 3‑D tensor
@triton.jit
def min_reduction_3d_kernel(
    x_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    L,  # length of the dimension to reduce
):
    out_idx = tl.program_id(0)

    # Compute the base offset and stride for the reduction dimension
    if dim == 0:
        i = out_idx // D2
        j = out_idx % D2
        base = i * D2 + j
        stride_reduce = D1 * D2
    elif dim == 1:
        k = out_idx // D2
        j = out_idx % D2
        base = k * D1 * D2 + j
        stride_reduce = D2
    else:  # dim == 2
        k = out_idx // D1
        i = out_idx % D1
        base = k * D1 * D2 + i * D2
        stride_reduce = 1

    # Initialise partial minima
    min_val = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)

    # Reduce over the dimension in chunks of BLOCK_SIZE
    offset = 0
    while offset < L:
        # Determine how many elements to load this iteration
        if offset + BLOCK_SIZE <= L:
            chunk_size = BLOCK_SIZE
        else:
            chunk_size = L - offset

        offsets = base + (offset + tl.arange(0, BLOCK_SIZE)) * stride_reduce
        mask = tl.arange(0, BLOCK_SIZE) < chunk_size
        vals = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
        min_val = tl.minimum(min_val, vals)

        offset += BLOCK_SIZE

    # Horizontal reduction of the partial minima
    for stride in [64, 32, 16, 8, 4, 2, 1]:
        if stride < BLOCK_SIZE:
            other = tl.shift(min_val, -stride)
            min_val = tl.minimum(min_val, other)

    out_val = min_val[0]
    tl.store(out_ptr + out_idx, out_val)


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the min of a 3‑D tensor over the specified dimension using the Triton kernel.
    """
    assert x.is_cuda
    x = x.contiguous()

    D0, D1, D2 = x.shape
    dim = dim % 3  # support negative indices
    L = [D0, D1, D2][dim]

    if dim == 0:
        out_shape = (D1, D2)
    elif dim == 1:
        out_shape = (D0, D2)
    else:
        out_shape = (D0, D1)

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    num_out = out.numel()
    BLOCK_SIZE = 128  # can be autotuned

    grid = lambda meta: (num