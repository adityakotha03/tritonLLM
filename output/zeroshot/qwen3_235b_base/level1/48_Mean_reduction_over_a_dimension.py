import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    reduction_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    if DIM == 0:
        offset_d0 = pid // (output_stride_0 * output_stride_1)
        offset_d1 = (pid % output_stride_0) // output_stride_1
        offset_d2 = pid % output_stride_1
        if offset_d0 >= reduction_size or offset_d1 >= output_stride_0 or offset_d2 >= output_stride_1:
            return
        block_start = offset_d0 * input_stride_0 + offset_d1 * input_stride_1 + offset_d2 * input_stride_2
    elif DIM == 1:
        offset_d0 = pid // (output_stride_0 * output_stride_1)
        offset_d1 = pid % output_stride_0
        offset_d2 = pid % output_stride_1
        if offset_d0 >= output_stride_0 or offset_d1 >= reduction_size or offset_d2 >= output_stride_1:
            return
        block_start = offset_d0 * input_stride_0 + offset_d1 * input_stride_1 + offset_d2 * input_stride_2
    else:  # DIM == 2
        offset_d0 = pid // (output_stride_0 * output_stride_1)
        offset_d1 = (pid % (output_stride_0 * output_stride_1)) // output_stride_1
        offset_d2 = pid % output_stride_1
        if offset_d0 >= output_stride_0 or offset_d1 >= output_stride_1 or offset_d2 >= reduction_size:
            return
        block_start = offset_d0 * input_stride_0 + offset_d1 * input_stride_1 + offset_d2 * input_stride_2

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    mask_base = tl.arange(0, BLOCK_SIZE)
    if DIM == 0:
        for i in range(reduction_size):
            offsets = block_start + i * input_stride_0 + mask_base * input_stride_2
            mask = (offset_d1 < input_stride_1) & (offset_d2 + mask_base < input_stride_2) & (mask_base < input_stride_2)
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            acc += x
        output_offset = offset_d1 * output_stride_1 + offset_d2
    elif DIM == 1:
        for i in range(reduction_size):
            offsets = block_start + i * input_stride_1 + mask_base * input_stride_2
            mask = (offset_d0 < input_stride_0) & (offset_d2 + mask_base < input_stride_2) & (mask_base < input_stride_2)
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            acc += x
        output_offset = offset_d0 * output_stride_1 + offset_d2
    else:  # DIM == 2
        for i in range(reduction_size):
            offsets = block_start + i * input_stride_2 + mask_base
            mask = (offset_d0 < input_stride_0) & (offset_d1 < input_stride_1) & (mask_base < input_stride_2)
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            acc += x
        output_offset = offset_d0 * output_stride_1 + offset_d1

    acc = acc.to(tl.float32)
    mean = acc / reduction_size
    output_offsets = output_offset + mask_base
    output_mask = mask_base == 0
    tl.store(output_ptr + output_offsets, tl.where(output_mask, mean[0], 0.0), mask=output_mask)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.contiguous()
    output_shape = list(x.shape)
    reduction_size = output_shape[dim]
    output_shape.pop(dim)
    out = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    if reduction_size == 0:
        return out

    input_strides = list(x.stride())
    output_strides = list(out.stride())

    def grid(meta):
        total_elements = out.numel()
        return (total_elements,)

    # Heuristic for block size
    BLOCK_SIZE = 1024
    while BLOCK_SIZE > reduction_size and BLOCK_SIZE > 1:
        BLOCK_SIZE //= 2

    mean_kernel[grid](
        x,
        out,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
        reduction_size,
        out.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        DIM=dim,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for mean reduction.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)