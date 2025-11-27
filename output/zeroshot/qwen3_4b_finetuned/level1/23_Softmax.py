import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1666677824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr0 + 393216 * x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (393216 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (786432 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (1179648 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = float('-inf')
    tmp4 = triton_helpers.maximum(tmp3, tmp5)
    tmp6 = triton_helpers.maximum(tmp4, tmp8)
    tmp7 = triton_helpers.maximum(tmp6, tmp11)
    tmp9 = triton_helpers.maximum(tmp7, tmp2)
    tmp10 = tmp1 - tmp9
    tmp12 = tl_math.exp(tmp10)
    tl.store(out_ptr0 + x2, tmp12, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1666677824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 393216 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (393216 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (786432 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1179648 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(1666677824)](arg0_1, buf0, 1666677824,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(1666677824)](buf0, buf1, 1666677824,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1,


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
