import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 8, 8192), (66048, 8192, 1))
    assert_size_stride(arg1_1, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 8, 8192), (66048, 8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(8192)](arg0_1, arg1_1, buf0, 8192, XBLOCK
            =128, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        del arg0_1
    return buf2, buf1, buf2, buf1,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.i2h = nn.Linear(1024 + 256, 256, bias=True)
        self.h2o = nn.Linear(256, 128, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]