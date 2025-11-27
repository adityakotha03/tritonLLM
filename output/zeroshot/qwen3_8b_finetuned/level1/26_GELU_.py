import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn.functional as F


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1600368640
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 393216
    x1 = xindex // 393216
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.full([XBLOCK, RBLOCK], 1.41421353553391, tl.float32)
    tmp2 = tmp0 / tmp1
    tmp3 = tl_math.erf(tmp2)
    tmp4 = 1.0 + tmp3
    tmp5 = 0.5 * tmp0
    tmp6 = tmp5 * tmp4
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty((4096, 393216), torch.float32, torch.strided_empty_strides)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(1600368640)](arg0_1, buf0, 1600368640,
            1, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]