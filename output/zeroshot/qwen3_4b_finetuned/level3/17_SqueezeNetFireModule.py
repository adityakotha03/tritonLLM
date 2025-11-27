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


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 6
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 6 * x2 + 54 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 6144 * y1), tmp0, ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 6
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 6 * x2 + 54 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 6
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, None)


@triton.jit
def triton_poi_fused_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, None)


@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 65536 % 128
    x0 = xindex % 65536
    x2 = xindex // 8388608
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 65536 * x2), tmp4, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + x1, tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp15 = tl.load(in_ptr2 + (x0 + 65536 * (-64 + x1) + 4194304 * x2),
        tmp12, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (-64 + x1), tmp12, eviction_policy=
        'evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp8, tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp12, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp11, tmp20)
    tl.store(out_ptr0 + x3, tmp21, None)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8) = args
    args.clear()
    assert_size_stride(primals_1, (6, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (6,), (1,))
    assert_size_stride(primals_3, (128, 3, 256, 256), (196608, 65536, 256, 1
        ))
    assert_size_stride(primals_4, (64, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_5, (64,), (1,))
    assert_size_stride(primals_6, (64, 6, 3, 3), (54, 9, 3, 1))
    assert_size_stride(primals_7, (64,), (1,))
    assert_size_stride(primals_8, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((6, 6, 1, 1), (6, 1, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(768, 9)](primals_1, buf0, 768, 9, XBLOCK=16,
            YBLOCK=64, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((128, 3, 256, 256), (196608, 1, 768, 3),
            torch.float32)
        triton_poi_fused_1[grid(12, 2048)](primals_3, buf1, 12, 2048,
            XBLOCK=64, YBLOCK=16, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((64, 6, 1, 1), (6, 1, 1, 1), torch.float32)
        triton_poi_fused_2[grid(384, 9)](primals_4, buf2, 384, 9, XBLOCK=16,
            YBLOCK=64, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((64, 6, 3, 3), (54, 1, 18, 6), torch.float32)
        triton_poi_fused_2[grid(384, 9)](primals_6, buf3, 384, 9, XBLOCK=16,
            YBLOCK=64, num_warps=4, num_stages=1)
        del primals_6
        buf4 = extern_kernels.convolution(buf1, buf0, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (128, 6, 256, 256), (393216, 1, 1536, 6))
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_relu_3[grid(5017600)](buf5, primals_2,
            5017600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf6 = extern_kernels.convolution(buf5, buf2, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (128, 64, 256, 256), (4194304, 1, 16384, 64))
        buf7 = buf6
        del buf6
        triton_poi_fused_convolution_relu_4[grid(536870912)](buf7, primals_5,
            536870912, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf8 = extern_kernels.convolution(buf5, buf3, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (128, 64, 256, 256), (4194304, 1, 16384, 64))
        buf9 = buf8
        del buf8
        triton_poi_fused_convolution_relu_4[grid(536870912)](buf9,
            primals_7, 536870912, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_7
        buf10 = empty_strided_cuda((128, 128, 256, 256), (8388608, 65536, 
            256, 1), torch.float32)
        triton_poi_fused_cat_5[grid(1073741824)](buf7, primals_8, buf9,
            primals_8, buf10, 1073741824, XBLOCK=512, num_warps=8, num_stages=1
            )
        del primals_8
    return buf10, buf0, buf2, buf3, buf5, buf7, buf9, primals_8


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(ModelNew, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, input_0):
        primals_1 = self.squeeze.weight
        primals_2 = self.squeeze.bias
        primals_4 = self.expand1x1.weight
        primals_5 = self.expand1x1.bias
        primals_6 = self.expand3x3.weight
        primals_7 = self.expand3x3.bias
        primals_8 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]
