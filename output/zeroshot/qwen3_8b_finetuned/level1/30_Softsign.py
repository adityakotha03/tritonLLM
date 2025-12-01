import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_div_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1610612096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


def triton_softsign(input_0):
    arg0_1 = input_0
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_abs_div_0[grid(1610612096)](buf1, arg0_1, 1610612096,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a Softsign activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = triton_softsign(arg0_1)
        return output