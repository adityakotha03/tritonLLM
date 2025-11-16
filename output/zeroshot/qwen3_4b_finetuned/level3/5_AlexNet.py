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
def triton_poi_fused__unsafe_index_convolution_relu_threshold_backward_0(
    in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 144 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_max_pool2d_with_indices_1(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1481760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x2 = xindex // 112 % 96
    x3 = xindex // 11616
    x4 = xindex // 10512 % 144
    x5 = xindex // 140736
    x1 = xindex // 10512
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 152736 * x3 + 21576 * x4),
        xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (112 + x0 + 576 * x1 + 152736 * x3 + 21576 * x4),
        xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (224 + x0 + 576 * x1 + 152736 * x3 + 21576 * x4),
        xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (10512 + x0 + 576 * x1 + 152736 * x3 +
        21576 * x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (10512 + x0 + 576 * x1 + 152736 * x3 +
        21576 * (x5 + 144)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tmp3 > tmp0
    tmp4 = tmp2 | tmp3
    tmp5 = tmp1 > tmp3
    tmp7 = tmp6 > tmp0
    tmp8 = tmp7 | tmp4
    tmp9 = tmp6 > tmp3
    tmp10 = tmp8 | tmp9
    tmp12 = tmp11 > tmp0
    tmp13 = tmp11 > tmp3
    tmp14 = tmp12 | tmp13
    tmp15 = tl.full([1], 1, tl.int8)
    tmp16 = tl.full([1], 0, tl.int8)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tl.store(out_ptr0 + (x2 + 96 * x0 + 144 * x1 + 10512 * x3), tmp17, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_2(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 224 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_max_pool2d_with_indices_3(in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1450448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 70
    x2 = xindex // 70 % 256
    x3 = xindex // 7776
    x4 = xindex // 7168 % 144
    x1 = xindex // 7168
    tmp0 = tl.load(in_ptr0 + (x0 + 480 * x1 + 384 * x3 + 4032 * x4),
        xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (70 + x0 + 480 * x1 + 384 * x3 + 4032 * x4),
        xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (140 + x0 + 480 * x1 + 384 * x3 + 4032 * x4),
        xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (7168 + x0 + 480 * x1 + 384 * x3 + 4032 * x4),
        xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (7168 + x0 + 480 * x1 + 384 * x3 +
        4032 * (x4 + 144)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tmp3 > tmp0
    tmp4 = tmp2 | tmp3
    tmp5 = tmp1 > tmp3
    tmp7 = tmp6 > tmp0
    tmp8 = tmp7 | tmp4
    tmp9 = tmp6 > tmp3
    tmp10 = tmp8 | tmp9
    tmp12 = tmp11 > tmp0
    tmp13 = tmp11 > tmp3
    tmp14 = tmp12 | tmp13
    tmp15 = tl.full([1], 1, tl.int8)
    tmp16 = tl.full([1], 0, tl.int8)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tl.store(out_ptr0 + (x2 + 256 * x0 + 144 * x1), tmp17, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_4(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 28737888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 70 % 384
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_5(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 57475776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 49 % 384
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_6(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 11297664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 36 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_7(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + 0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + 0)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + 0)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl_math.abs(tmp4)
    tmp15 = triton_helpers.maximum(tmp14, tmp8)
    tmp16 = triton_helpers.maximum(tmp15, tmp12)
    tmp17 = tl_math.abs(tmp1)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 - tmp18
    tmp20 = tl_math.abs(tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp18 * tmp18
    tmp23 = tmp21 / tmp22
    tmp24 = tmp23 + tmp0
    tmp25 = tmp24 > tmp1
    tl.store(out_ptr0 + x0, tmp25, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_8(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 1024
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_9(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 1024
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_10(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23) = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_2, (96,), (1,))
    assert_size_stride(primals_3, (1024, 3, 224, 224), (150528, 50176, 224,
        1))
    assert_size_stride(primals_4, (256, 96, 5, 5), (2400, 25, 5, 1))
    assert_size_stride(primals_5, (256,), (1,))
    assert_size_stride(primals_6, (384, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_7, (384,), (1,))
    assert_size_stride(primals_8, (384, 384, 3, 3), (4161, 9, 3, 1))
    assert_size_stride(primals_9, (384,), (1,))
    assert_size_stride(primals_10, (256, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_11, (256,), (1,))
    assert_size_stride(primals_12, (4096, 256, 6, 6), (9216, 36, 6, 1))
    assert_size_stride(primals_13, (4096,), (1,))
    assert_size_stride(primals_14, (4096, 4096, 1, 1), (16384, 1, 1, 1))
    assert_size_stride(primals_15, (4096,), (1,))
    assert_size_stride(primals_16, (1000, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_17, (1000,), (1,))
    assert_size_stride(primals_18, (96, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_19, (96,), (1,))
    assert_size_stride(primals_20, (256, 96, 5, 5), (2400, 25, 5, 1))
    assert_size_stride(primals_21, (256,), (1,))
    assert_size_stride(primals_22, (384, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_23, (384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(4,),
            padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1024, 96, 56, 56), (271360, 2800, 56, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((1024, 96, 56, 56), (271360, 2800, 56, 1),
            torch.bool)
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_convolution_relu_threshold_backward_0[
            grid(12582912)](buf1, primals_2, buf2, 12582912, XBLOCK=512,
            num_warps=8, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((1024, 96, 56, 56), (271360, 2800, 56, 1),
            torch.bool)
        triton_poi_fused__unsafe_index_max_pool2d_with_indices_1[grid(1481760)](
            buf1, buf3, 1481760, XBLOCK=1024, num_warps=4, num_stages=1)
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf1, (1024, 96,
            56, 56), (271360, 2800, 56, 1), 0), primals_4, stride=(1, 1),
            padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (1024, 256, 56, 56), (73728, 2800, 56, 1))
        buf5 = buf4
        del buf4
        buf6 = empty_strided_cuda((1024, 256, 56, 56), (73728, 2800, 56, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_2[grid(12582912)](
            buf5, primals_5, buf6, 12582912, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_5
        buf7 = empty_strided_cuda((1024, 256, 56, 56), (73728, 2800, 56, 1),
            torch.bool)
        triton_poi_fused__unsafe_index_max_pool2d_with_indices_3[grid(1450448)](
            buf5, buf7, 1450448, XBLOCK=1024, num_warps=4, num_stages=1)
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf5, (1024, 256,
            56, 56), (73728, 2800, 56, 1), 0), primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (1024, 384, 56, 56), (110592, 2800, 56, 1))
        buf9 = buf8
        del buf8
        buf10 = empty_strided_cuda((1024, 384, 56, 56), (110592, 2800, 56, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_4[grid(28737888)](
            buf9, primals_7, buf10, 28737888, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_7
        buf11 = extern_kernels.convolution(buf9, primals_8, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (1024, 384, 56, 56), (110592, 2800, 56, 1))
        buf12 = buf11
        del buf11
        buf13 = empty_strided_cuda((1024, 384, 56, 56), (110592, 2800, 56, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_4[grid(28737888)](
            buf12, primals_9, buf13, 28737888, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_9
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf12, (1024,
            384, 56, 56), (110592, 2800, 56, 1), 0), primals_10, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (1024, 256, 56, 56), (73728, 2800, 56, 1))
        buf15 = buf14
        del buf14
        buf16 = empty_strided_cuda((1024, 256, 56, 56), (73728, 2800, 56, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_5[grid(57475776)](
            buf15, primals_11, buf16, 57475776, XBLOCK=256, num_warps=16,
            num_stages=1)
        del primals_11
        buf17 = extern_kernels.convolution(buf15, primals_12, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (1024, 256, 27, 27), (186624, 729, 27, 1))
        buf18 = buf17
        del buf17
        buf19 = empty_strided_cuda((1024, 256, 27, 27), (186624, 729, 27, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_6[grid(11297664)](
            buf18, primals_13, buf19, 11297664, XBLOCK=256, num_warps=16,
            num_stages=1)
        del primals_13
        buf20 = empty_strided_cuda((1024, 256, 27, 27), (186624, 729, 27, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_7[grid(1024)](
            primals_14, buf20, 1024, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_14
        buf21 = extern_kernels.convolution(reinterpret_tensor(buf18, (1024,
            256, 27, 27), (186624, 729, 27, 1), 0), primals_15, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (1024, 4096, 27, 27), (2434736, 9216, 27, 1
            ))
        buf22 = buf21
        del buf21
        buf23 = empty_strided_cuda((1024, 4096, 27, 27), (2434736, 9216, 27,
            1), torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_8[grid(4194304)](
            buf22, primals_16, buf23, 4194304, XBLOCK=512, num_warps=8,
            num_stages=1)
        del primals_16
        buf24 = extern_kernels.convolution(reinterpret_tensor(buf22, (1024,
            4096, 27, 27), (2434736, 9216, 27, 1), 0), primals_17, stride=(1,
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (1024, 4096, 27, 27), (2434736, 9216, 27, 1
            ))
        buf25 = buf24
        del buf24
        buf26 = empty_strided_cuda((1024, 4096, 27, 27), (2434736, 9216, 27,
            1), torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_9[grid(4194304)](
            buf25, primals_18, buf26, 4194304, XBLOCK=512, num_warps=8,
            num_stages=1)
        del primals_18
        buf27 = empty_strided_cuda((1000, 1024), (1024, 1), torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_10[grid(1000)](
            primals_19, buf27, 1000, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_19
        buf28 = extern_kernels.convolution(reinterpret_tensor(buf25, (1024,
            4096, 27, 27), (2434736, 9216, 27, 1), 0), primals_20, stride=(1,
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (1024, 256, 27, 27), (186624, 729, 27, 1))
        buf29 = buf28
        del buf28
        buf30 = empty_strided_cuda((1024, 256, 27, 27), (186624, 729, 27, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_6[grid(11297664)](
            buf29, primals_21, buf30, 11297664, XBLOCK=256, num_warps=16,
            num_stages=1)
        del primals_21
        buf31 = extern_kernels.convolution(buf29, primals_22, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (1024, 384, 27, 27), (291600, 729, 27, 1))
        buf32 = buf31
        del buf31
        buf33 = empty_strided_cuda((1024, 384, 27, 27), (291600, 729, 27, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_4[grid(28737888)](
            buf32, primals_23, buf33, 28737888, XBLOCK=256, num_warps=16,
            num_stages=1)
        del primals_23
        buf34 = extern_kernels.convolution(buf32, primals_23, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (1024, 384, 27, 27), (291600, 729, 27, 1))
        buf35 = buf34
        del buf34
        buf36 = empty_strided_cuda((1024, 384, 27, 27), (291600, 729, 27, 1),
            torch.bool)
        triton_poi_fused_convolution_relu_threshold_backward_4[grid(28737888)](
            buf35, primals_23, buf36, 28737