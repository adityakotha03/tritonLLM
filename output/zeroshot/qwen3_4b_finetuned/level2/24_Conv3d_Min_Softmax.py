import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 24
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 98304
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 98304 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 98304 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 98304 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tl.store(out_ptr0 + x3, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 98304
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 98304 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 98304 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 98304 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 / tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (24,), (1,))
    assert_size_stride(primals_3, (128, 3, 24, 32, 32), (73728, 24576, 1024,
        32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            2, 3], stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
            transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = reinterpret_tensor(buf1, (128, 24, 32, 32, 24), (589824, 24576,
            768, 24, 1), 0)
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(165888)](buf2, primals_2, 
            165888, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 24, 32, 32, 24), (589824, 1, 18364,
            580, 24), torch.float32)
        triton_poi_fused__softmax_1[grid(165888)](buf2, buf3, 165888,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = reinterpret_tensor(buf2, (128, 24, 32, 32, 24), (589824, 1,
            18364, 580, 24), 0)
        del buf2
        triton_poi_fused__softmax_2[grid(165888)](buf3, buf4, 165888,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf3
    return buf4, primals_1, primals_3


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
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
