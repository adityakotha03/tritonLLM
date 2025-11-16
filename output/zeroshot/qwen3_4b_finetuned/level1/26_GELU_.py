import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1589259072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 > 0.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp2, tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp6 + tmp2
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 <= tmp7
    tmp10 = tl.where(tmp9, tmp7, tmp8)
    tmp11 = tmp5 + tmp10
    tmp12 = 0.7978845608028654
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + x2, tmp13, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(1589259072)](arg0_1, buf0, 1589259072,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
