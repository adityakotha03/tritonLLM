import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 393216
    x1 = xindex // 393216
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 393216 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (393216 + x0 + 393216 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (786432 + x0 + 393216 * x1), xmask)
    tmp6 = tl.load(in_ptr0 + (1179648 + x0 + 393216 * x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp6
    tmp7 = tmp0 - tmp5
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp1 - tmp5
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp3 - tmp5
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 + tmp13
    tmp15 = tmp6 - tmp5
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 + tmp16
    tl.store(out_ptr0 + x2, tmp17, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(16777216)](arg0_1, buf0, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
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
