import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_hardtanh_hardtanh_backward_0(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 0.5
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 + tmp3
    tmp9 = tmp8 * tmp6
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 0.0
    tmp12 = tmp10 <= tmp11
    tmp13 = tmp10 >= tmp3
    tmp14 = tmp12 | tmp13
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        buf1 = empty_strided_cuda((4096, 393216), (393216, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_hardtanh_backward_0[grid(1572864)](arg0_1,
            buf0, buf1, 1572864, XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
    return buf0, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
