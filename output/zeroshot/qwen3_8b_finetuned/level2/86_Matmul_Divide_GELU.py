import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = 10.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + x0, tmp2, None)


@triton.jit
def triton_poi_fused_gelu_1(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, None)
    tmp1 = 0.5
    tmp3 = tl.load(in_out_ptr0 + x2, None)
    tmp4 = tmp3 * tmp1
    tmp5 = tl_math.erf(tmp0 / 1.4142135623731)
    tmp6 = 1.0 + tmp5
    tmp7 = tmp4 * tmp6
    tl.store(in_out_ptr0 + x0, tmp7, None)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_div_0[grid(8192)](primals_1, buf2, 8192, XBLOCK=256,
            num_warps=4, num_stages=1)
        del primals_1
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        triton_poi_fused_gelu_1[grid(8192)](buf4, 8192, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf4
        del primals_2
    return buf0, buf1, buf3, buf4


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8192, 8192, bias=True)

    def forward(self, input_0):
        primals_2 = self.linear.bias
        primals_1 = self.linear.weight
        primals_2 = primals_2
        primals_1 = primals_1
        output = call([input_0, primals_1, primals_2])
        return output[0]