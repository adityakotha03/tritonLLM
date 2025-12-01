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
def triton_poi_fused_min_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = xindex // 24
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2 + x2), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.minimum(tmp1, tmp2)
    tmp4 = triton_helpers.minimum(tmp0, tmp3)
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 24 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (12 + x0 + 24 * x2), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (24 + x0 + 24 * x2), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (36 + x0 + 24 * x2), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (48 + x0 + 24 * x2), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (60 + x0 + 24 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tl.store(out_ptr0 + x2, tmp13, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 3, 24, 32, 32), (73728, 24576, 9216,
        288, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 24, 24, 32, 32), (221184, 9216, 384,
            12, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_min_0[grid(36864)](primals_2, buf0, 36864, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((128, 24, 32, 32), (24576, 9216, 288, 9),
            torch.float32)
        triton_poi_fused__softmax_1[grid(36864)](buf0, buf1, 36864, XBLOCK=
            128, num_warps=1, num_stages=1)
        buf2 = buf0
        del buf0
    return buf1, primals_1, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]