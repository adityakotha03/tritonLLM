import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 - tmp1
    tmp3 = 0.2
    tmp4 = tmp2 - tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_mish_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 256, 256), (524288, 65536, 256, 1
        ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 3, 3), (589824, 1, 196608, 65536
            ), torch.float32)
        buf1 = reinterpret_tensor(buf0, (128, 64, 3, 3), (589824, 1, 196608,
            65536), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(131072)](buf1, primals_1, 
            131072, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((128, 64, 256, 256), (4194304, 1, 16384,
            64), torch.float32)
        triton_poi_fused_add_sub_1[grid(131072)](buf1, buf2, 131072, XBLOCK
            =512, num_warps=8, num_stages=1)
        buf3 = empty_strided_cuda((128, 64, 256, 256), (4194304, 1, 16384,
            64), torch.float32)
        triton_poi_fused_mish_2[grid(131072)](buf2, buf3, 131072, XBLOCK=
            512, num_warps=8, num_stages=1)
    return buf3, primals_2, primals_3, buf1, buf2


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
