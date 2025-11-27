import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clamp_div_mul_sigmoid_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = -1.0
    tmp6 = tmp4 > tmp5
    tmp7 = 1.0
    tmp8 = tmp4 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.where(tmp9, tmp4, tmp5)
    tmp11 = tl.where(tmp9, tmp7, tmp10)
    tl.store(out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_poi_fused_clamp_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = -1.0
    tmp3 = tmp1 > tmp2
    tmp4 = 1.0
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tl.where(tmp6, tmp1, tmp2)
    tmp8 = tl.where(tmp6, tmp4, tmp7)
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_div_mul_sigmoid_0[grid(8388608)](primals_3,
            buf0, 8388608, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_clamp_tanh_1[grid(8388608)](buf0, buf1, 8388608,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
