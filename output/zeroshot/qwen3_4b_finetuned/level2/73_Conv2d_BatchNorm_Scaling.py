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
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 8 * x2 + 72 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 128 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8 * x2 + 1024 * y1), tmp0, xmask & ymask)


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
    y0 = yindex % 64
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 64 * x2 + 576 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 128 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 64 * x2 + 8192 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * x2 + 576 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 9 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_native_batch_norm_5(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (32768 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (65536 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (98304 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)
    tl.store(out_ptr2 + x0, tmp23, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp3 = tl.load(in_ptr3 + x0, xmask)
    tmp4 = tl.load(in_ptr4 + x0, xmask)
    tmp5 = tmp1 * tmp2
    tmp6 = tmp0 - tmp3
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 * tmp4
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_mul_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 128
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x2 + 8192 * x1), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128, 
        1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 8, 3, 3), (72, 1, 24, 8), torch.float32
            )
        get_raw_stream(0)
        triton_poi_fused_0[grid(512, 9)](primals_1, buf0, 512, 9, XBLOCK=16,
            YBLOCK=64, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((128, 8, 128, 128), (131072, 1, 1024, 8),
            torch.float32)
        triton_poi_fused_1[grid(512, 128)](primals_3, buf1, 512, 128,
            XBLOCK=64, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.
            float32)
        triton_poi_fused_2[grid(4096, 9)](primals_2, buf2, 4096, 9, XBLOCK=
            16, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 64, 128, 128), (1048576, 1, 8192, 
            64), torch.float32)
        triton_poi_fused_3[grid(8192, 128)](primals_4, buf3, 8192, 128,
            XBLOCK=32, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((128, 64, 124, 124), (98304, 1, 792, 6),
            torch.float32)
        triton_poi_fused_convolution_4[grid(32768, 9)](buf1, buf0, buf4, 
            32768, 9, XBLOCK=16, YBLOCK=256, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((1, 64, 128, 128), (1048576, 16384, 128, 
            1), torch.float32)
        buf6 = empty_strided_cuda((1, 64, 128, 128), (1048576, 16384, 128, 
            1), torch.float32)
        buf7 = empty_strided_cuda((1, 64, 128, 128), (1048576, 16384, 128, 
            1), torch.float32)
        triton_poi_fused_native_batch_norm_5[grid(32768)](buf4, buf5, buf6,
            buf7, 32768, XBLOCK=512, num_warps=8, num_stages=1)
        buf8 = empty_strided_cuda((128, 64, 124, 124), (98304, 1536, 124, 1
            ), torch.float32)
        triton_poi_fused_native_batch_norm_6[grid(32768)](buf4, buf5, buf6,
            buf7, primals_5, buf8, 32768, XBLOCK=512, num_warps=8, num_stages=1
            )
        del buf5
        del buf6
        del buf7
        del primals_5
        buf9 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        triton_poi_fused_mul_7[grid(2097152)](buf8, buf9, 2097152, XBLOCK=
            1024, num_warps=4, num_stages=1)
    return buf9, buf0, buf2, buf3, buf4, buf8


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bn.weight
        primals_5 = self.bn.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
