import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_0(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 2
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r2, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r2 + 64 * x0), xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + r2, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp2 - tmp16
    tmp24 = 64.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tl.store(in_out_ptr0 + (r2 + 64 * x3), tmp6, xmask)
    tl.store(out_ptr2 + x3, tmp28, xmask)
    tl.store(out_ptr0 + x3, tmp16, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_1(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 2
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r2, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r2 + 64 * x0), xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + r2, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp2 - tmp16
    tmp24 = 64.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tl.store(in_out_ptr0 + (r2 + 64 * x3), tmp6, xmask)
    tl.store(out_ptr2 + x3, tmp28, xmask)
    tl.store(out_ptr0 + x3, tmp16, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_2(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 2
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r2, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r2 + 64 * x0), xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + r2, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp2 - tmp16
    tmp24 = 64.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tl.store(in_out_ptr0 + (r2 + 64 * x3), tmp6, xmask)
    tl.store(out_ptr2 + x3, tmp28, xmask)
    tl.store(out_ptr0 + x3, tmp16, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_3(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 2
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r2, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r2 + 64 * x0), xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + r2, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp2 - tmp16
    tmp24 = 64.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tl.store(in_out_ptr0 + (r2 + 64 * x3), tmp6, xmask)
    tl.store(out_ptr2 + x3, tmp28, xmask)
    tl.store(out_ptr0 + x3, tmp16, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_4(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 2
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r2, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r2 + 64 * x0), xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + r2, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp2 - tmp16
    tmp24 = 64.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tl.store(in_out_ptr0 + (r2 + 64 * x3), tmp6, xmask)
    tl.store(out_ptr2 + x3, tmp28, xmask)
    tl.store(out_ptr0 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused__to_copy_5(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.001
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32, primals_33
        ) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64,), (1,))
    assert_size_stride(primals_6, (2, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64,), (1,))
    assert_size_stride(primals_9, (64,), (1,))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (64,), (1,))
    assert_size_stride(primals_12, (64,), (1,))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_14, (64,), (1,))
    assert_size_stride(primals_15, (64,), (1,))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (64,), (1,))
    assert_size_stride(primals_18, (64,), (1,))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64,), (1,))
    assert_size_stride(primals_21, (64,), (1,))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64,), (1,))
    assert_size_stride(primals_24, (64,), (1,))
    assert_size_stride(primals_25, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_26, (64,), (1,))
    assert_size_stride(primals_27, (64,), (1,))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64,), (1,))
    assert_size_stride(primals_30, (64,), (1,))
    assert_size_stride(primals_31, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_32, (64,), (1,))
    assert_size_stride(primals_33, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf1 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf2 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf3 = reinterpret_tensor(buf2, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf2
        buf4 = reinterpret_tensor(buf1, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf1
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_relu_0[grid(128)](
            buf3, primals_1, primals_6, primals_2, buf0, buf4, 128, 64,
            XBLOCK=8, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf5 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf6 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf7 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf8 = reinterpret_tensor(buf7, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf7
        buf9 = reinterpret_tensor(buf6, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf6
        triton_per_fused__native_batch_norm_legit_convolution_relu_1[grid(128)](
            buf8, primals_3, primals_4, primals_5, buf5, buf9, 128, 64,
            XBLOCK=8, num_warps=4, num_stages=1)
        del primals_3
        del primals_4
        del primals_5
        buf10 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf11 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf12 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf13 = reinterpret_tensor(buf12, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf12
        buf14 = reinterpret_tensor(buf11, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf11
        triton_per_fused__native_batch_norm_legit_convolution_relu_2[grid(128)](
            buf13, primals_7, primals_8, primals_9, buf10, buf14, 128, 64,
            XBLOCK=8, num_warps=4, num_stages=1)
        del primals_7
        del primals_8
        del primals_9
        buf15 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf16 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf17 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf18 = reinterpret_tensor(buf17, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf17
        buf19 = reinterpret_tensor(buf16, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf16
        triton_per_fused__native_batch_norm_legit_convolution_relu_3[grid(128)](
            buf18, primals_10, primals_11, primals_12, buf15, buf19, 128, 
            64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_10
        del primals_11
        del primals_12
        buf20 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf21 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf22 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf23 = reinterpret_tensor(buf22, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf22
        buf24 = reinterpret_tensor(buf21, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf21
        triton_per_fused__native_batch_norm_legit_convolution_relu_4[grid(128)](
            buf23, primals_13, primals_14, primals_15, buf20, buf24, 128, 
            64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_13
        del primals_14
        del primals_15
        buf25 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf26 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf27 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf28 = reinterpret_tensor(buf27, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf27
        buf29 = reinterpret_tensor(buf26, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf26
        triton_per_fused__native_batch_norm_legit_convolution_relu_0[grid(128)](
            buf28, primals_16, primals_17, primals_18, buf25, buf29, 128, 
            64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_16
        del primals_17
        del primals_18
        buf30 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf31 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf32 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf33 = reinterpret_tensor(buf32, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf32
        buf34 = reinterpret_tensor(buf31, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf31
        triton_per_fused__native_batch_norm_legit_convolution_relu_1[grid(128)](
            buf33, primals_19, primals_20, primals_21, buf30, buf34, 128, 
            64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_19
        del primals_20
        del primals_21
        buf35 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf36 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf37 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf38 = reinterpret_tensor(buf37, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf37
        buf39 = reinterpret_tensor(buf36, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf36
        triton_per_fused__native_batch_norm_legit_convolution_relu_2[grid(128)](
            buf38, primals_22, primals_23, primals_24, buf35, buf39, 128, 
            64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_22
        del primals_23
        del primals_24
        buf40 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf41 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf42 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf43 = reinterpret_tensor(buf42, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf42
        buf44 = reinterpret_tensor(buf41, (2, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf41
        triton_per_fused__native_batch_norm_legit_convolution_relu_3[grid(128)](
            buf43, primals_25, primals_26, primals_27, buf40, buf44, 128, 
            64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_25
        del primals_26
        del primals_27
        buf45 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf46 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf47 = empty_strided_cuda((2, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf48 = reinterpret_tensor(buf47, (2, 64, 1, 1), (64, 1, 64, 64), 0)
       