import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_1(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp8, xmask)


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
        triton_poi_fused_mul_0[grid(8388608)](primals_3, buf0, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mul_0[grid(67108864)](primals_1, buf1, 67108864,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mul_0[grid(8388608)](buf0, buf2, 8388608, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_leaky_relu_leaky_relu_backward_1[grid(8388608)](buf2,
            buf1, buf3, buf1, 8388608, XBLOCK=512, num_warps=8, num_stages=1)
        del buf1
    return buf3, primals_2, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
