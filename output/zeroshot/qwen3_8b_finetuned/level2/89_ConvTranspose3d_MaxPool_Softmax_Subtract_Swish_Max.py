import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_max_pool2_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 32 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (16 + x2 + 32 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + x2 + 32 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (48 + x2 + 32 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp0 > tmp1
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.full([1, 1], 1, tl.int64)
    tmp10 = tl.where(tmp7, tmp9, tmp8)
    tmp11 = tmp1 > tmp0
    tmp12 = tl.full([1, 1], 2, tl.int64)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp3 > tmp2
    tmp15 = tl.full([1, 1], 3, tl.int64)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tmp17 = tmp5 > tmp4
    tmp18 = tl.full([1, 1], 4, tl.int64)
    tmp19 = tl.where(tmp17, tmp18, tmp16)
    tl.store(out_ptr0 + (x2 + 32 * y3), tmp6, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 32 * y3), tmp19, xmask & ymask)


@triton.jit
def triton_poi_fused_max_pool2_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + (16 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (48 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp4 = tmp3 > tmp1
    tmp6 = tmp5 > tmp3
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp8, xmask & ymask)
    tmp9 = tl.load(in_ptr0 + (16 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp11 = tmp9 > tmp10
    tmp12 = tmp5 > tmp9
    tmp13 = tmp12 > tmp11
    tl.store(out_ptr1 + (x2 + 16 * y3), tmp13, xmask & ymask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 64 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (32 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (64 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (96 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tmp9 / tmp1
    tmp11 = tmp10 / tmp2
    tmp12 = tmp11 / tmp4
    tmp13 = tmp12 / tmp6
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_sub_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_max_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * tmp2
    tmp5 = tl.load(in_ptr0 + (16 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (16 + y0), ymask, eviction_policy='evict_last')
    tmp7 = tmp5 - tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp8 * tmp7
    tmp10 = triton_helpers.maximum(tmp4, tmp9)
    tmp11 = tl.load(in_ptr0 + (32 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (32 + y0), ymask, eviction_policy='evict_last')
    tmp13 = tmp11 - tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp14 * tmp13
    tmp16 = triton_helpers.maximum(tmp10, tmp15)
    tmp17 = tl.load(in_ptr0 + (48 + x2 + 16 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (48 + y0), ymask, eviction_policy='evict_last')
    tmp19 = tmp17 - tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp20 * tmp19
    tmp22 = triton_helpers.maximum(tmp16, tmp21)
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp22, xmask & ymask)


def call(args):
    (primals_1, primals_2, primals_3) = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 3, 16, 32, 32), (16384, 5461, 341,
        10.625, 1))
    assert_size_stride(primals_3, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 19, 34, 34), (16384, 1024, 54.4,
            1.5972222222222222, 1))
        buf1 = empty_strided_cuda((128, 16, 9, 17, 17), (16384, 1024, 179.2,
            10.588235294117647, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 16, 9, 17, 17), (16384, 1024, 179.2,
            10.588235294117647, 1), torch.int64)
        get_raw_stream(0)
        triton_poi_fused_max_pool2_with_indices_0[grid(16, 16)](buf0, buf1,
            buf2, 16, 16, XBLOCK=16, YBLOCK=16, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((128, 16, 9, 17, 17), (16384, 1024, 179.2,
            10.588235294117647, 1), torch.int64)
        triton_poi_fused_max_pool2_with_indices_1[grid(16, 16)](buf1, buf3,
            buf2, 16, 16, XBLOCK=16, YBLOCK=16, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 16, 9, 17, 17), (16384, 1024, 179.2,
            10.588235294117647, 1), torch.float32)
        triton_poi_fused__softmax_2[grid(16384)](buf2, buf4, 16384, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((128, 16, 9, 17, 17), (16384, 1024, 179.2,
            10.588235294117647, 1), torch.float32)
        triton_poi_fused_sub_3[grid(16, 16)](buf4, buf5, 16, 16, XBLOCK=16,
            YBLOCK=16, num_warps=4, num_stages=1)
        buf6 = reinterpret_tensor(buf4, (128, 9, 17, 17), (16384, 179.2,
            10.588235294117647, 1), 0)
        del buf4
        triton_poi_fused_max_4[grid(16, 16)](buf5, buf6, 16, 16, XBLOCK=16,
            YBLOCK=16, num_warps=4, num_stages=1)
        del buf5
    return (buf6, primals_1, primals_2, primals_3, buf1, buf2, buf3)


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
        primals_3 = self.subtract
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]