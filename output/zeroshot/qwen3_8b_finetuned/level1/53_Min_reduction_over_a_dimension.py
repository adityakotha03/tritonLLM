import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_min_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl
    .constexpr):
    xnumel = 131072
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    xmask = xindex < xnumel
    x2 = xindex
    x3 = xindex
    x0 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK, 1])
    tmp4 = tl.broadcast_to(tmp1, [1, XBLOCK])
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tl.store(in_out_ptr0 + x0, tmp5, xmask)
    tmp6 = tl.load(in_ptr0 + x1, xmask)
    tmp7 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp8 = tl.broadcast_to(tmp6, [XBLOCK, 1])
    tmp10 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp11 = triton_helpers.maximum(tmp8, tmp10)
    tl.store(in_out_ptr0 + x1, tmp11, xmask)
    tmp12 = tl.load(in_ptr0 + x0, xmask)
    tmp13 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp14 = tl.broadcast_to(tmp12, [XBLOCK, 1])
    tmp16 = tl.broadcast_to(tmp13, [1, XBLOCK])
    tmp17 = triton_helpers.maximum(tmp14, tmp16)
    tl.store(in_out_ptr0 + x0, tmp17, xmask)
    tmp18 = tl.load(in_ptr0 + x0, xmask)
    tmp19 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp19, [1, XBLOCK])
    tmp23 = triton_helpers.maximum(tmp20, tmp22)
    tl.store(in_out_ptr0 + x0, tmp23, xmask)


def triton_min(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: 
    tl.constexpr):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert_size_stride(in_out_ptr0, (128, 4095), (4095, 1))
    assert_size_stride(in_ptr0, (128, 4096, 4095), (16777216, 4096, 1))
    assert_size_stride(in_ptr1, (128, 4095, 4096), (16777216, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4095), (4095, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_min_0[grid(131072)](in_out_ptr0, in_ptr0, in_ptr1,
            131072, 1, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
    return out_ptr0


class ModelNew(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        arg0_2 = input_0
        arg0_3 = input_0
        output = triton_min(arg0_1, arg0_2, arg0_3, arg0_1, 131072, 1, XBLOCK=
            128)
        return output