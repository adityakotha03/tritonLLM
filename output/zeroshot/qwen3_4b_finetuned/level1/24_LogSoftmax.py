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
def triton_poi_fused__log_softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 157527936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 393216 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (393248 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (786496 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp0 - tmp5
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused__log_softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 157527936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 393216 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (393248 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (786496 + 393216 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp2 + tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp9 = tl_math.log(tmp8)
    tmp10 = tmp0 - tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__log_softmax_0[grid(157527936)](arg0_1, buf0, 
            157527936, XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        triton_poi_fused__log_softmax_1[grid(157527936)](buf0, buf1, 
            157527936, XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
