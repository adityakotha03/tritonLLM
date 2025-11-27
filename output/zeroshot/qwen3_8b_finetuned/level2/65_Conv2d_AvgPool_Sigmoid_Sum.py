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
def triton_poi_fused_add_relu_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    xbatch = xindex // 589824
    x2 = xindex % 589824
    x0 = xindex % 256
    x1 = xindex // 256
    x4 = xbatch
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = -tmp0
    tmp2 = tl.full([1], 1.0, tl.int32)
    tmp3 = tl.full([1], 0.0, tl.int32)
    tmp4 = tl.exp(tmp1)
    tmp5 = tmp2 + tmp4
    tmp6 = 1.0 / tmp5
    tmp7 = tl.full([1], 0.0, tl.int32)
    tmp8 = tmp7 + tmp6
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tl.store(out_ptr0 + x3, tmp9, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 96, 96), (589824, 9216, 96, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 96, 96), (589824, 9216, 96, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_relu_sum_0[grid(589824)](arg0_1, buf0, 589824,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(8, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.avg_pool = nn.AvgPool2d((4, 4), stride=(4, 4), padding=(0, 0))

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]