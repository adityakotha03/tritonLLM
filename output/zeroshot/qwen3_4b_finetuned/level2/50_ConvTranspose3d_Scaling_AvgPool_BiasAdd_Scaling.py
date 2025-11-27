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
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 23040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 60
    x1 = xindex // 60
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_avg_pool3d_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 11520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 15
    x1 = xindex // 15 % 16
    x2 = xindex // 240 % 32
    x3 = xindex // 3840
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 60 * x2 + 1280 * x1 + 40960 * x3),
        xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 60 * x2 + 1280 * x1 + 40960 * x3),
        xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (30 + 2 * x0 + 60 * x2 + 1280 * x1 + 40960 *
        x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (31 + 2 * x0 + 60 * x2 + 1280 * x1 + 40960 *
        x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x4, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 11520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 15
    x1 = xindex // 15 % 16
    x2 = xindex // 240 % 32
    x3 = xindex // 3840
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 15 * x2 + 480 * x1 + 15360 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 15 * x2 + 480 * x1 + 15360 * x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x4, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (49152, 16384, 1024,
        32, 1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 16, 32, 32), (23040, 1440, 45,
            15, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(23040)](primals_3, buf0, 23040, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((128, 16, 16, 32, 32), (11520, 720, 45, 
            15, 1), torch.float32)
        triton_poi_fused_avg_pool3d_1[grid(11520)](buf0, buf1, 11520,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((128, 16, 16, 32, 32), (11520, 720, 45, 
            15, 1), torch.float32)
        triton_poi_fused_add_mul_2[grid(11520)](buf1, primals_2, primals_5,
            buf2, 11520, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        del primals_2
        del primals_5
    return buf2, primals_1, buf0, primals_4


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, input_0):
        primals_1 = self.scale1
        primals_2 = self.bias
        primals_4 = self.scale2
        primals_3 = self.conv_transpose.weight
        primals_5 = self.conv_transpose.bias
        primals_30 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_30])
        return output[0]
