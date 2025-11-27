import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_0[ext_fn](buf0, arg1_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_1[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_2[ext_fn](buf0, arg1_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_3[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_5(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_4[ext_fn](buf0, arg1_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_5[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_7(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_6[ext_fn](buf0, arg1_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_7[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_9(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_8[ext_fn](buf0, arg1_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_9[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_11(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_10[ext_fn](buf0, arg1_1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_11[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_13(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_12[ext_fn](buf0, arg1_1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_13[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_15(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_14[ext_fn](buf0, arg1_1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_15[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(arg0_1, (128, 128, 4, 4), (2048, 1, 512,
        128), 0)


@triton.jit
def triton_poi_fused_convolution_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4096 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_min_mul_17(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp7 * tmp7
    tmp11 = 0.0625
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 256, 64),
            torch.float32)
        triton_poi_fused_convolution_16[ext_fn](buf0, arg1_1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 128, 4, 4), (2048, 1, 512, 128),
            torch.float32)
        triton_poi_fused_add_gelu_min_mul_17[ext_fn](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=8