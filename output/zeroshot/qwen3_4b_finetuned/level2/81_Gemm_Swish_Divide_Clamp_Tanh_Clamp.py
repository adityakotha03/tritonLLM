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
def triton_poi_fused_add_div_mul_sigmoid_tanh_clamp_0(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 / tmp3
    tmp5 = -1.0
    tmp6 = tl.where(xmask, tmp4, tmp5)
    tmp7 = tmp6 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp10 < tmp9
    tmp12 = tmp7 > tmp8
    tmp13 = tmp11 & tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp8)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp15 + tmp8
    tmp17 = tmp16 > tmp15
    tmp18 = tl.where(tmp17, tmp14, tmp15)
    tmp19 = tl.full(tmp18.shape, -1.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp19, tmp18)
    tl.store(out_ptr0 + x0, tmp20, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_mul_sigmoid_tanh_clamp_0[grid(1024)](arg0_1,
            buf0, 1024, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input_0):
        arg0_1 = self.gemm.weight
        arg0_2 = self.gemm.bias
        output = call([arg0_1, arg0_2])
        return output[0]
