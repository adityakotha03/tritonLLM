import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F


@triton.jit
def triton_poi_fused_sigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -tmp0
    tmp2 = tl_math.exp(tmp1)
    tmp3 = tl_math.add(tmp2, 1)
    tmp4 = tl_math.div(1, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 32768), (32768, 1))
    assert_size_stride(arg1_1, (32768, 32768), (32768, 1))
    assert_size_stride(arg2_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.mm(arg0_1, arg1_1.t()) + arg2_1
        buf1 = empty((128, 32768), torch.float32, torch.cuda.defaultStream(0))
        get_raw_stream(0)
        triton_poi_fused_sigmoid_0[grid(4194304)](buf0, buf1, 4194304, XBLOCK=128)
        del buf0
        buf2 = torch.mm(buf1, torch.ones((32768, 1), dtype=torch.float32, device='cuda'))
    return buf2, arg0_1, arg1_1, arg2_1


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32768, 32768)

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = self.linear.weight
        arg2_1 = self.linear.bias
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]