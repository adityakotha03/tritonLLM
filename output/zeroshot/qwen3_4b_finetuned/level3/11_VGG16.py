import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 81216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2256 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 81216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2256 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 784 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 784 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_4(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 784 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 49 % 512
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 49 % 512
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_7(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 49 % 512
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_8(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1000
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tl.store(in_out_ptr0 + x2, tmp5, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_9(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1000
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tl.store(in_out_ptr0 + x2, tmp5, xmask)
    tl.store(out_ptr0 + x2, tmp3, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32,
        primals_33, primals_34, primals_35, primals_36, primals_37,
        primals_38, primals_39, primals_40, primals_41) = args
    args.clear()
    assert_size_stride(primals_1, (64, 6, 6, 6), (216, 36, 6, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (10, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (10,), (1,))
    assert_size_stride(primals_5, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (64,), (1,))
    assert_size_stride(primals_7, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (128,), (1,))
    assert_size_stride(primals_9, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_10, (128,), (1,))
    assert_size_stride(primals_11, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_12, (256,), (1,))
    assert_size_stride(primals_13, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_14, (256,), (1,))
    assert_size_stride(primals_15, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_16, (256,), (1,))
    assert_size_stride(primals_17, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_18, (512,), (1,))
    assert_size_stride(primals_19, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_20, (512,), (1,))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_22, (512,), (1,))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_24, (512,), (1,))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_26, (512,), (1,))
    assert_size_stride(primals_27, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_28, (512,), (1,))
    assert_size_stride(primals_29, (4096, 512), (512, 1))
    assert_size_stride(primals_30, (4096,), (1,))
    assert_size_stride(primals_31, (4096, 4096), (4096, 1))
    assert_size_stride(primals_32, (4096,), (1,))
    assert_size_stride(primals_33, (1000, 4096), (4096, 1))
    assert_size_stride(primals_34, (1000,), (1,))
    assert_size_stride(primals_35, (1000, 1000), (1000, 1))
    assert_size_stride(primals_36, (1000,), (1,))
    assert_size_stride(primals_37, (1000, 1000), (1000, 1))
    assert_size_stride(primals_38, (1000,), (1,))
    assert_size_stride(primals_39, (1000, 1000), (1000, 1))
    assert_size_stride(primals_40, (1000,), (1,))
    assert_size_stride(primals_41, (1000, 1000), (1000, 1))
    assert_size_stride(primals_42, (1000,), (1,))
    assert_size_stride(primals_43, (1000, 1000), (1000, 1))
    assert_size_stride(primals_44, (1000,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (10, 64, 224, 224), (32512, 512, 224, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(81216)](buf1, primals_4,
            buf0, 81216, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_4
        buf2 = extern_kernels.convolution(buf1, primals_3, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (10, 64, 224, 224), (32512, 512, 224, 1))
        buf3 = buf2
        del buf2
        triton_poi_fused_convolution_relu_1[grid(81216)](buf3, primals_6,
            buf1, 81216, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_6
        buf4 = extern_kernels.convolution(buf3, primals_5, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (10, 128, 112, 112), (16384, 128, 112, 1))
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_relu_2[grid(32512)](buf5, primals_8,
            buf3, 32512, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_8
        buf6 = extern_kernels.convolution(buf5, primals_7, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (10, 128, 112, 112), (16384, 128, 112, 1))
        buf7 = buf6
        del buf6
        triton_poi_fused_convolution_relu_3[grid(32512)](buf7, primals_10,
            buf5, 32512, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_10
        buf8 = extern_kernels.convolution(buf7, primals_9, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (10, 256, 56, 56), (65536, 256, 56, 1))
        buf9 = buf8
        del buf8
        triton_poi_fused_convolution_relu_4[grid(32512)](buf9, primals_12,
            buf7, 32512, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_12
        buf10 = extern_kernels.convolution(buf9, primals_13, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (10, 256, 56, 56), (73728, 256, 56, 1))
        buf11 = buf10
        del buf10
        triton_poi_fused_convolution_relu_2[grid(32512)](buf11, primals_14,
            buf9, 32512, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_14
        buf12 = extern_kernels.convolution(buf11, primals_15, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (10, 256, 56, 56), (73728, 256, 56, 1))
        buf13 = buf12
        del buf12
        triton_poi_fused_convolution_relu_3[grid(32512)](buf13, primals_16,
            buf11, 32512, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_16
        buf14 = extern_kernels.convolution(buf13, primals_17, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (10, 512, 28, 28), (393216, 512, 28, 1))
        buf15 = buf14
        del buf14
        triton_poi_fused_convolution_relu_4[grid(16384)](buf15, primals_18,
            buf13, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_18
        buf16 = extern_kernels.convolution(buf15, primals_19, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (10, 512, 28, 28), (409600, 512, 28, 1))
        buf17 = buf16
        del buf16
        triton_poi_fused_convolution_relu_5[grid(16384)](buf17, primals_20,
            buf15, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_20
        buf18 = extern_kernels.convolution(buf17, primals_21, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (10, 512, 28, 28), (409600, 512, 28, 1))
        buf19 = buf18
        del buf18
        triton_poi_fused_convolution_relu_3[grid(16384)](buf19, primals_22,
            buf17, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_22
        buf20 = extern_kernels.convolution(buf19, primals_23, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (10, 512, 14, 14), (98304, 512, 14, 1))
        buf21 = buf20
        del buf20
        triton_poi_fused_convolution_relu_4[grid(49504)](buf21, primals_24,
            buf19, 49504, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_24
        buf22 = extern_kernels.convolution(buf21, primals_25, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (10, 512, 14, 14), (98304, 512, 14, 1))
        buf23 = buf22
        del buf22
        triton_poi_fused_convolution_relu_5[grid(49504)](buf23, primals_26,
            buf21, 49504, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_26
        buf24 = extern_kernels.convolution(buf23, primals_27, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (10, 512, 14, 14), (98304, 512, 14, 1))
        buf25 = buf24
        del buf24
        triton_poi_fused_convolution_relu_3[grid(49504)](buf25, primals_28,
            buf23, 49504, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_28
        buf26 = extern_kernels.convolution(buf25, primals_29, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (10, 512, 7, 7), (25088, 512, 7, 1))
        buf27 = buf26
        del buf26
        triton_poi_fused_convolution_relu_4[grid(16384)](buf27, primals_30,
            buf25, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_30
        buf28 = extern_kernels.convolution(buf27, primals_31, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (10, 4096, 7, 7), (196608, 4096, 7, 1))
        buf29 = buf28
        del buf28
        triton_poi_fused_convolution_relu_6[grid(16384)](buf29, primals_32,
            buf27, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_32
        buf30 = extern_kernels.convolution(buf29, primals_33, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (10, 1000, 7, 7), (49000, 1000, 7, 1))
        buf31 = buf30
        del buf30
        triton_poi_fused_convolution_relu_7[grid(10000)](buf31, primals_34,
            buf29, 10000, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_34
        buf32 = extern_kernels.convolution(buf31, primals_35, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (10, 1000, 1, 1), (1000, 1000, 1, 1))
        buf33 = buf32
        del buf32
        triton_poi_fused_convolution_relu_8[grid(1000)](buf33, primals_36,
            buf31, 1000, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_36
        buf34 = extern_kernels.convolution(buf33, primals_37, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (10, 1000, 1, 1), (1000, 1000, 1, 1))
        buf35 = buf34
        del buf34
        triton_poi_fused_convolution_relu_9[grid(1000)](buf35, primals_38,
            buf33, 1000, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_38
        buf36 = extern_kernels.convolution(buf35, primals_39, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (10, 1000, 1, 1), (1000, 1000, 1, 1))
        buf37 = buf36
        del buf36
        triton_poi_fused_convolution_relu_9[grid(1000)](buf37, primals_40,
            buf35, 1000, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_40
        buf38 = extern_kernels.convolution(buf37, primals_41, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (10, 1000, 1, 1), (1000, 1000, 1, 1))
        buf39 = buf38
        del buf38
        triton_poi_fused_convolution_relu_9[grid(1000)](buf39, primals_42,
            buf37, 1000, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_42
        buf40 = empty_strided_cuda((10, 1000), (1000, 1), torch.float32)
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf39, (10, 1000
            ), (1000, 1), 0), primals_44, alpha=1, beta=1, out=buf40)
        del primals_44
        buf41 = empty_strided_cuda((10, 1000), (1000, 1), torch.float32)
        extern_kernels.addmm(primals_45, buf40, primals_46, alpha=1, beta=1,
            out=buf41)
        del primals_45
        del primals_46
    return (buf39, buf37, buf35, buf33, primals_1, primals_3, primals_5,
        primals_7, primals_9, primals_11, primals_13, primals_15,
        primals_17, primals_19, primals_21, primals_23, primals_25,
        primals_27, primals_29, primals_31, primals_33, primals_35,
        primals_37, primals_39, primals_41, buf40, buf41, primals_43)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet