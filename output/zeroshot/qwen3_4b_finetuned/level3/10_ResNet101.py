import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 196416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 64 * x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 196416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 32256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 196416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 128 * x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 14400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 44064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 256 * x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 29160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 86784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 24032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 512 * x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 190608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_12(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 190608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 168128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 512 * x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 549648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 9
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_15(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 549648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32,
        primals_33, primals_34, primals_35, primals_36, primals_37,
        primals_38, primals_39) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_8, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_9, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_10, (2048,), (1,))
    assert_size_stride(primals_11, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_12, (2048,), (1,))
    assert_size_stride(primals_13, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_14, (2048,), (1,))
    assert_size_stride(primals_15, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_16, (2048,), (1,))
    assert_size_stride(primals_17, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_18, (2048,), (1,))
    assert_size_stride(primals_19, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_20, (2048,), (1,))
    assert_size_stride(primals_21, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_22, (2048,), (1,))
    assert_size_stride(primals_23, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_24, (2048,), (1,))
    assert_size_stride(primals_25, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_26, (2048,), (1,))
    assert_size_stride(primals_27, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_28, (2048,), (1,))
    assert_size_stride(primals_29, (1000,), (1,))
    assert_size_stride(primals_30, (1000,), (1,))
    assert_size_stride(primals_31, (1000,), (1,))
    assert_size_stride(primals_32, (1000,), (1,))
    assert_size_stride(primals_33, (1000,), (1,))
    assert_size_stride(primals_34, (1000,), (1,))
    assert_size_stride(primals_35, (1000,), (1,))
    assert_size_stride(primals_36, (1000,), (1,))
    assert_size_stride(primals_37, (1000,), (1,))
    assert_size_stride(primals_38, (1000,), (1,))
    assert_size_stride(primals_39, (1000,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 64, 7, 7), (3136, 49, 7, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(196416)](primals_1, primals_2, buf0,
            196416, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        del primals_1
        buf1 = extern_kernels.convolution(buf0, primals_4, stride=(2, 2),
            padding=(3, 3), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (10, 64, 224, 224), (32256, 512, 224, 1))
        buf2 = buf1
        del buf1
        triton_poi_fused_convolution_relu_1[grid(196416)](buf2, primals_3,
            196416, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
        buf3 = empty_strided_cuda((10, 64, 112, 112), (784, 12, 1, 1),
            torch.float32)
        extern_kernels.max_pool2d_with_indices(buf2, [3, 3], [2, 2], [1, 1],
            [0, 0])
        buf4 = extern_kernels.convolution(buf3, primals_5, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (10, 64, 112, 112), (8192, 128, 1, 1))
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_relu_1[grid(32256)](buf5, primals_6, 
            32256, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_6
        buf6 = extern_kernels.convolution(buf5, primals_7, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (10, 256, 112, 112), (32256, 128, 112, 1))
        buf7 = buf6
        del buf6
        triton_poi_fused_convolution_2[grid(32256)](buf7, primals_8, 32256,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_8
        buf8 = extern_kernels.convolution(buf7, primals_9, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (10, 1024, 112, 112), (125440, 1024, 112,
            1))
        buf9 = extern_kernels.convolution(buf8, primals_10, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (10, 2048, 112, 112), (240320, 1024, 112, 
            1))
        buf10 = empty_strided_cuda((10, 1024, 112, 112), (125440, 128, 112,
            1), torch.float32)
        triton_poi_fused_clone_4[grid(24576)](primals_10, primals_11, buf10,
            24576, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_10
        buf11 = extern_kernels.convolution(buf10, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (10, 256, 113, 113), (312720, 128, 113, 
            1))
        buf12 = buf11
        del buf11
        triton_poi_fused_convolution_5[grid(14400)](buf12, primals_13, 14400,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_13
        buf13 = extern_kernels.convolution(buf12, primals_14, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (10, 512, 113, 113), (645920, 128, 113, 
            1))
        buf14 = buf13
        del buf13
        triton_poi_fused_convolution_6[grid(44064)](buf14, primals_15, 44064,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_15
        buf15 = empty_strided_cuda((10, 1024, 113, 113), (125440, 128, 113,
            1), torch.float32)
        triton_poi_fused_clone_7[grid(49152)](primals_16, primals_17, buf15,
            49152, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_16
        del primals_17
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (10, 512, 114, 114), (645920, 128, 114, 1))
        buf17 = buf16
        del buf16
        triton_poi_fused_convolution_8[grid(29160)](buf17, primals_19, 29160
            , XBLOCK=256, num_warps=4, num_stages=1)
        del primals_19
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (10, 1024, 114, 114), (1321280, 128, 114,
            1))
        buf19 = buf18
        del buf18
        triton_poi_fused_convolution_9[grid(86784)](buf19, primals_21, 86784,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_21
        buf20 = empty_strided_cuda((10, 1024, 114, 114), (1321280, 128, 114,
            1), torch.float32)
        triton_poi_fused_clone_10[grid(24032)](primals_22, primals_23, buf20,
            24032, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_22
        del primals_23
        buf21 = extern_kernels.convolution(buf20, primals_24, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (10, 512, 115, 115), (649736, 128, 115, 1))
        buf22 = buf21
        del buf21
        triton_poi_fused_convolution_11[grid(190608)](buf22, primals_25, 190608
            , XBLOCK=256, num_warps=4, num_stages=1)
        del primals_25
        buf23 = extern_kernels.convolution(buf22, primals_26, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (10, 2048, 115, 115), (2320224, 128, 115,
            1))
        buf24 = buf23
        del buf23
        triton_poi_fused_convolution_relu_12[grid(190608)](buf24, primals_27,
            190608, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_27
        buf25 = empty_strided_cuda((10, 1024, 115, 115), (1321280, 128, 115,
            1), torch.float32)
        triton_poi_fused_clone_13[grid(168128)](primals_28, primals_29, buf25,
            168128, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_28
        del primals_29
        buf26 = extern_kernels.convolution(buf25, primals_30, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (10, 512, 116, 116), (655840, 128, 116, 1))
        buf27 = buf26
        del buf26
        triton_poi_fused_convolution_14[grid(549648)](buf27, primals_31, 549648
            , XBLOCK=256, num_warps=4, num_stages=1)
        del primals_31
        buf28 = extern_kernels.convolution(buf27, primals_32, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (10, 2048, 116, 116), (2389408, 128, 116,
            1))
        buf29 = buf28
        del buf28
        triton_poi_fused_convolution_relu_15[grid(549648)](buf29, primals_33,
            549648, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_33
        buf30 = empty_strided_cuda((10, 1024, 116, 116), (1321280, 128, 116,
            1), torch.float32)
        triton_poi_fused_clone_13[grid(168128)](primals_34, primals_35, buf30,
            168128, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_34
        del primals_35
        buf31 = extern_kernels.convolution(buf30, primals_36, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (10, 512, 117, 117), (655840, 128, 117, 1))
        buf32 = buf31
        del buf31
        triton_poi_fused_convolution_14[grid(549648)](buf32, primals_37, 549648
            , XBLOCK=256, num_warps=4, num_stages=1)
        del primals_37
        buf33 = extern_kernels.convolution(buf32, primals_38, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (10, 2048, 117, 117), (2389408, 128, 117,
            1))
        buf34 = buf33
        del buf33
        triton_poi_fused_convolution_relu_15[grid(549648)](buf34, primals_39,
            549648, XBLOCK=256, num_warps=4, num_stages=1)
        del primals