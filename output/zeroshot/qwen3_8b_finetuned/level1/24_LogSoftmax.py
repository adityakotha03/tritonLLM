import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_log_softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 159253248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 393216
    x1 = xindex // 393216
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0 + 196608), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (x1 + 196608), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tmp6 = tmp3 - tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp9 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp10 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tmp7 + tmp13
    tmp15 = tl_math.log(tmp14)
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_log_softmax_0[grid(159253248)](arg0_1, buf0, 
            159253248, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


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