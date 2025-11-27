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
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 96
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 48 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 32
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 8 * x2 + 128 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 8
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2
    y1 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 32 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 2 * x2 + 64 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 16 * x2 + 144 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 16 * x2 + 144 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 32
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 32 * x2 + 288 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 32
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 32 * x2 + 288 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 96
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 96 * x2 + 864 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 96
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 96 * x2 + 864 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 160
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 160 * x2 + 1440 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 160
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 160 * x2 + 1440 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 320
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 320 * x2 + 2880 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
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
    y0 = yindex % 320
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 320 * x2 + 2880 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_15(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 96
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.full([1, 1], 1, tl.int32)
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 3 * x2 + 48 * y1), tmp2, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 3 * x2 + 48 * y1), tmp6, xmask & ymask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_16(in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 96
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, YBLOCK])
    tl.where(ymask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, YBLOCK])
    tmp8 = tl.where(ymask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, YBLOCK])
    tmp17 = tl.where(ymask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 16.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(out_ptr2 + (x2 + 16 * y3), tmp19, xmask & ymask)
    tl.store(out_ptr3 + (x2 + 16 * y3), tmp24, xmask & ymask)
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 16 * y3), tmp12, xmask & ymask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_17(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 16 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 16 * y3), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + y0, None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + y0, None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + y0, None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = 16.0
    tmp8 = tmp5 / tmp6
    tmp9 = 1e-05
    tmp10 = tmp7 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp0 * tmp11
    tmp13 = tl.full([1, 1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp8, xmask)
    tl.store(out_ptr1 + (x2 + 16 * y3), tmp11, xmask)
    tl.store(out_ptr2 + (x2 + 16 * y3), tmp14, xmask)


@triton.jit
def triton_poi_fused__to_copy_18(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_19(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_relu_20(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8,
    in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4,
    out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11,
    out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17,
    out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23,
    out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29,
    out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35,
    out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41,
    out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46, out_ptr47,
    out_ptr48, out_ptr49, out_ptr50, out_ptr51, out_ptr52, out_ptr53,
    out_ptr54, out_ptr55, out_ptr56, out_ptr57, out_ptr58, out_ptr59,
    out_ptr60, out_ptr61, out_ptr62, out_ptr63, out_ptr64, out_ptr65,
    out_ptr66, out_ptr67, out_ptr68, out_ptr69, out_ptr70, out_ptr71,
    out_ptr72, out_ptr73, out_ptr74, out_ptr75, out_ptr76, out_ptr77,
    out_ptr78, out_ptr79, out_ptr80, out_ptr81, out_ptr82, out_ptr83,
    out_ptr84, out_ptr85, out_ptr86, out_ptr87, out_ptr88, out_ptr89,
    out_ptr90, out_ptr91, out_ptr92, out_ptr93, out_ptr94, out_ptr95,
    out_ptr96, out_ptr97, out_ptr98, out_ptr99, out_ptr100, out_ptr101,
    out_ptr102, out_ptr103, out_ptr104, out_ptr105, out_ptr106, out_ptr107,
    out_ptr108, out_ptr109, out_ptr110, out_ptr111, out_ptr112, out_ptr113,
    out_ptr114, out_ptr115, out_ptr116, out_ptr117, out_ptr118, out_ptr119,
    out_ptr120, out_ptr121, out_ptr122, out_ptr123, out_ptr124, out_ptr125,
    out_ptr126, out_ptr127, out_ptr128, out_ptr129, out_ptr130, out_ptr131,
    out_ptr132, out_ptr133, out_ptr134, out_ptr135, out_ptr136, out_ptr137,
    out_ptr138, out_ptr139, out_ptr140, out_ptr141, out_ptr142, out_ptr143,
    out_ptr144, out_ptr145, out_ptr146, out_ptr147, out_ptr148, out_ptr149,
    out_ptr150, out_ptr151, out_ptr152, out_ptr153, out_ptr154, out_ptr155,
    out_ptr156, out_ptr157, out_ptr158, out_ptr159, out_ptr160, out_ptr161,
    out_ptr162, out_ptr163, out_ptr164, out_ptr165, out_ptr166, out_ptr167,
    out_ptr168, out_ptr169, out_ptr170, out_ptr171, out_ptr172, out_ptr173,
    out_ptr174, out_ptr175, out_ptr176, out_ptr177, out_ptr178, out_ptr179,
    out_ptr180, out_ptr181, out_ptr182, out_ptr183, out_ptr184, out_ptr185,
    out_ptr186, out_ptr187, out_ptr188, out_ptr189, out_ptr190, out_ptr191,
    out_ptr192, out_ptr193, out_ptr194, out_ptr195, out_ptr196, out_ptr197,
    out_ptr198, out_ptr199, out_ptr200, out_ptr201, out_ptr202, out_ptr203,
    out_ptr204, out_ptr205, out_ptr206, out_ptr207, out_ptr208, out_ptr209,
    out_ptr210, out_ptr211, out_ptr212, out_ptr213, out_ptr214, out_ptr215,
    out_ptr216, out_ptr217, out_ptr218, out_ptr219, out_ptr220, out_ptr221,
    out_ptr222, out_ptr223, out_ptr224, out_ptr225, out_ptr226, out_ptr227,
    out_ptr228, out_ptr229, out_ptr230, out_ptr231, out_ptr232, out_ptr233,
    out_ptr234, out_ptr235, out_ptr236, out_ptr237, out_ptr238, out_ptr239,
    out_ptr240, out_ptr241, out_ptr242, out_ptr243, out_ptr244, out_ptr245,
    out_ptr246, out_ptr247, out_ptr248, out_ptr249, out_ptr250, out_ptr251,
    out_ptr252, out_ptr253, out_ptr254, out_ptr255, out_ptr256, out_ptr257,
    out_ptr258, out_ptr259, out_ptr260, out_ptr261, out_ptr262, out_ptr263,
    out_ptr264, out_ptr265, out_ptr266, out_ptr267, out_ptr268, out_ptr269,
    out_ptr270, out_ptr271, out_ptr272, out_ptr273, out_ptr274, out_ptr275,
    out_ptr276, out_ptr277, out_ptr278, out_ptr279, out_ptr280, out_ptr281,
    out_ptr282, out_ptr283, out_ptr284, out_ptr285, out_ptr286, out_ptr287,
    out_ptr288, out_ptr289, out_ptr290, out_ptr291, out_ptr292, out_ptr293,
    out_ptr294, out_ptr295, out_ptr296, out_ptr297, out_ptr298, out_ptr299,
    out_ptr300, out_ptr301, out_ptr302, out_ptr303, out_ptr304, out_ptr305,
    out_ptr306, out_ptr307, out_ptr308, out_ptr309, out_ptr310, out_ptr311,
    out_ptr312, out_ptr313, out_ptr314, out_ptr315, out_ptr316, out_ptr317,
    out_ptr318, out_ptr319, out_ptr320, out_ptr321, out_ptr322, out_ptr323,
    out_ptr324, out_ptr325, out_ptr326, out_ptr327, out_ptr328, out_ptr329,
    out_ptr330, out_ptr331, out_ptr332, out_ptr333, out_ptr334, out_ptr335,
    out_ptr336, out_ptr337, out_ptr338, out_ptr339, out_ptr340, out_ptr341,
    out_ptr342, out_ptr343, out_ptr344, out_ptr345, out_ptr346, out_ptr347,
    out_ptr348, out_ptr349, out_ptr350, out_ptr351, out_ptr352, out_ptr353,
    out_ptr354, out_ptr355, out_ptr356, out_ptr357, out_ptr358, out_ptr359,
    out_ptr360, out_ptr361, out_ptr362, out_ptr363, out_ptr364, out_ptr365,
    out_ptr366, out_ptr367, out_ptr368, out_ptr369, out_ptr370, out_ptr371,
    out_ptr372, out_ptr373, out_ptr374, out_ptr375, out_ptr376, out_ptr377,
    out_ptr378, out_ptr379, out_ptr380, out_ptr381, out_ptr382, out_ptr383,
    out_ptr384, out_ptr385, out_ptr386, out_ptr387, out_ptr388, out_ptr389,
    out_ptr390, out_ptr391, out_ptr392, out_ptr393, out_ptr394, out_ptr395,
    out_ptr396, out_ptr397, out_ptr398, out_ptr399, out_ptr400, out_ptr401,
    out_ptr402, out_ptr403, out_ptr404, out_ptr405, out_ptr406, out_ptr407,
   