import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_mean_div_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 65535
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x2 = x2 % xnumel
    x2 = x2 // rnumel
    x2 = x2 % rnumel
    x2 = x2 + tl.program_id(0) * rnumel
    x2mask = x2 < xnumel
    x0 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (x0, x1), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(x2mask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 65535.0
    tmp14 = tmp12 / tmp13
    tl.store(out_ptr0 + x2, tmp14, x2mask)


@triton.jit
def triton_poi_fused_abs_mean_div_1(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 65535
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x2 = x2 % xnumel
    x2 = x2 // rnumel
    x2 = x2 % rnumel
    x2 = x2 + tl.program_id(0) * rnumel
    x2mask = x2 < xnumel
    x0 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (x0, x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x2, x2mask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + x2, tmp2, x2mask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 65535), (65535, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 65535), (65535, 1), torch.float32)
        buf1 = empty_strided_cuda((32768, 65535), (65535, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_abs_mean_div_0[grid(65535)](arg0_1, buf0, 65535, 32768,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        triton_poi_fused_abs_mean_div_1[grid(65535)](buf0, arg0_1, buf1,
            65535, 32768, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf1,


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]