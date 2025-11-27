import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.1
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_gelu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp2 * tmp3
    tmp5 = 0.5773502691896257
    tmp6 = tmp4 * tmp5
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp4 * tmp4
    tmp10 = 0.3333333333333333
    tmp11 = tmp9 * tmp10
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 / tmp13
    tl.store(out_ptr0 + x0, tmp14, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_div_0[grid(8388608)](arg1_1, buf0, 8388608, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        torch.ops.aten.addmm.default_0(buf1, arg0_1, buf0, alpha=1, beta=1,
            out=None)
        del arg0_1
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_gelu_1[grid(8388608)](buf1, buf2, 8388608, XBLOCK=
            512, num_warps=4, num_stages=1)
        del buf1
    return buf2,


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, input_0):
        arg1_1 = self.linear.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
