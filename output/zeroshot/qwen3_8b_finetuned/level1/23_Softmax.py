import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 159258048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x2 - 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (1024 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1024 * x1 + 1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (1024 * x1 + 512), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (1024 * x1 + 513), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (1024 * x1 + 1023), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp5, tmp7)
    tmp8 = triton_helpers.maximum(tmp6, tmp10)
    tmp9 = triton_helpers.maximum(tmp8, tmp13)
    tmp11 = tmp0 - tmp9
    tmp12 = tl_math.exp(tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 + tmp14
    tmp16 = tmp4 + tmp15
    tmp17 = tmp12 + tmp16
    tmp18 = tmp7 + tmp17
    tmp19 = tmp12 + tmp18
    tmp20 = tmp10 + tmp19
    tmp21 = tmp12 + tmp20
    tmp22 = tmp12 / tmp21
    tl.store(out_ptr0 + x2, tmp22, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(159258048)](arg0_1, buf0, 
            159258048, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]