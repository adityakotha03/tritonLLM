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
    xnumel = 516064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 13824 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 16
    x2 = xindex // 512 % 8
    x3 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 128 * x1 + 2048 * x2 + 8192 * x3), 
        xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 128 * x1 + 2048 * x2 + 8192 * x3
        ), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2 * x0 + 128 * x1 + 2048 * x2 + 8192 *
        x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2 * x0 + 128 * x1 + 2048 * x2 + 8192 *
        x3), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl_math.abs(tmp1)
    tmp9 = tl_math.abs(tmp0)
    tmp10 = tmp8 > tmp9
    tmp11 = tmp7 | tmp10
    tmp12 = tmp3 > tmp2
    tmp13 = tl_math.abs(tmp3)
    tmp14 = tl_math.abs(tmp2)
    tmp15 = tmp13 > tmp14
    tmp16 = tmp12 | tmp15
    tmp17 = tmp11 | tmp16
    tmp18 = tmp5 > tmp4
    tmp19 = tl_math.abs(tmp5)
    tmp20 = tl_math.abs(tmp4)
    tmp21 = tmp19 > tmp20
    tmp22 = tmp18 | tmp21
    tmp23 = tmp17 | tmp22
    tl.store(out_ptr0 + x4, tmp6, xmask)
    tl.store(out_ptr1 + x4, tmp23, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_sub_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_mul_sigmoid_sub_5(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp2 * tmp0
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_max_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (49152, 16384, 1024,
        32, 1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(reinterpret_tensor(
            primals_3, (1, 3, 16, 32, 32), (16384, 5120, 320, 10, 1), 0),
            primals_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1,
            1), transposed=True, output_padding=(1, 1, 1), groups=1,
            bias=None)
        buf1 = buf0
        del buf0
        buf2 = reinterpret_tensor(buf1, (128, 16, 16, 32, 32), (262144, 
            16384, 1024, 32, 1), 0)
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(516064)](buf2, primals_2, 
            516064, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 16, 8, 16, 16), (20480, 1280, 160,
            10, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 16, 8, 16, 16), (20480, 1280, 160,
            10, 1), torch.bool)
        triton_poi_fused_max_pool3d_with_indices_1[grid(131072)](buf2, buf3,
            buf4, 131072, XBLOCK=512, num_warps=8, num_stages=1)
        buf5 = reinterpret_tensor(buf2, (128, 16, 8, 16, 16), (20480, 1280,
            160, 10, 1), 0)
        del buf2
        triton_poi_fused__softmax_2[grid(131072)](buf3, buf5, 131072,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf6 = reinterpret_tensor(buf3, (128, 16, 8, 16, 16), (20480, 1280,
            160, 10, 1), 0)
        del buf3
        triton_poi_fused__softmax_3[grid(131072)](buf5, buf6, 131072,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf7 = reinterpret_tensor(buf5, (128, 16, 8, 16, 16), (20480, 1280,
            160, 10, 1), 0)
        del buf5
        triton_poi_fused_sub_4[grid(131072)](buf6, primals_4, buf7, 131072,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_4
        buf8 = reinterpret_tensor(buf6, (128, 16, 8, 16, 16), (20480, 1280,
            160, 10, 1), 0)
        del buf6
        triton_poi_fused_mul_sigmoid_sub_5[grid(131072)](buf7, buf6, buf8,
            131072, XBLOCK=512, num_warps=8, num_stages=1)
        buf9 = reinterpret_tensor(buf4, (128, 16, 8, 16, 16), (20480, 1280,
            160, 10, 1), 0)
        del buf4
        triton_poi_fused_max_6[grid(131072)](buf8, buf9, 131072, XBLOCK=
            512, num_warps=8, num_stages=1)
        del buf8
    return buf9, primals_1, primals_3, primals_5, buf7


class ModelNew(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels)) # Assuming subtraction is element-wise across channels

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.subtract
        primals_5 = input_0
        primals_3 = self.max_pool.weight
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
