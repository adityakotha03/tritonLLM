import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1593199616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = 1.0
    tmp3 = 1.4142135623730957
    tmp4 = tmp0 / tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = tmp5 + tmp2
    tmp7 = tmp0 * tmp6
    tmp8 = tmp1 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(1593199616)](arg0_1, buf0, 1593199616,
            XBLOCK=1024, num_warps=4, num_stages=1)
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