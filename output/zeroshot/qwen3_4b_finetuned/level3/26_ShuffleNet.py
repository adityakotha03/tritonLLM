import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 768
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = yindex // 24
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 24 * x2 + 168 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 7 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 720
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = yindex // 240
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 240 * x2 + 720 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 3 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 480
    y1 = yindex // 480
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 480 * x2 + 1440 * y1), xmask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 3 * y3), tmp0, xmask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 960
    y1 = yindex // 960
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 960 * x2 + 2880 * y1), xmask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 3 * y3), tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_4(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 24
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3
    y1 = yindex // 3
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3 * x2 + 150528 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 50176 * y3), tmp4, xmask & ymask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 62500
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = xindex // 112
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 448 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 448 * x1), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (224 + 2 * x0 + 448 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (225 + 2 * x0 + 448 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 240
    xnumel = 11264
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3
    y1 = yindex // 3
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3 * x2 + 33792 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 11264 * y3), tmp4, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_relu_7(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 24
    xnumel = 11264
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3
    y1 = yindex // 3
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3 * x2 + 33792 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 11264 * y3), tmp4, xmask & ymask)


@triton.jit
def triton_poi_fused__to_copy_8(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 < tmp4
    tmp6 = tmp1 >= tmp4
    tmp7 = tl.full([1], 1, tl.int8)
    tmp8 = tl.full([1], 0, tl.int8)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp5 & tmp9
    tmp11 = tmp10.to(tl.int64)
    tl.store(out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 100
    xnumel = 2880
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 3
    x2 = xindex // 3
    y0 = yindex % 100
    y1 = yindex // 100
    x3 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 100 * x2 + 28800 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + 1440 * y1), ymask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr4 + x3, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0 + 100 * x2 + 28800 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2 + 1440 * y1), ymask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr7 + x3, xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0 + 100 * x2 + 28800 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 - tmp5
    tmp11 = tmp10 + tmp1
    tmp12 = tmp11 + tmp3
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 + tmp16
    tmp17 = tmp15 - tmp13
    tmp18 = tmp9 * tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 + tmp3
    tmp22 = tmp21 + tmp13
    tmp23 = tmp22 + tmp16
    tmp24 = tmp23 - tmp13
    tmp25 = tmp18 * tmp24
    tl.store(out_ptr0 + (x3 + 2880 * y4), tmp25, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 100 % 100
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_12(in_ptr0, in_ptr1, out_ptr0,
    ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1000
    y1 = yindex // 1000
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1000 * x2 + 16000 * y1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_13(in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr):
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1000
    y1 = yindex // 1000
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1000 * x2 + 16000 * y1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp4, xmask)
    tl.store(out_ptr1 + (y0 + 1000 * x2 + 16000 * y1), tmp6, xmask)


@triton.jit
def triton_poi_fused_add_14(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_out_ptr0 + x0, xmask)
    tmp2 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused__to_copy_15(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 < tmp4
    tmp6 = tmp1 >= tmp4
    tmp7 = tl.full([1], 1, tl.int8)
    tmp8 = tl.full([1], 0, tl.int8)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp5 & tmp9
    tmp11 = tmp10.to(tl.int64)
    tl.store(out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 100
    xnumel = 2880
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 3
    x2 = xindex // 3
    y0 = yindex % 100
    y1 = yindex // 100
    x3 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 100 * x2 + 28800 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + 1440 * y1), ymask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr4 + x3, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0 + 100 * x2 + 28800 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2 + 1440 * y1), ymask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr7 + x3, xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0 + 100 * x2 + 28800 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 - tmp5
    tmp11 = tmp10 + tmp1
    tmp12 = tmp11 + tmp3
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 + tmp16
    tmp17 = tmp15 - tmp13
    tmp18 = tmp9 * tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 + tmp3
    tmp22 = tmp21 + tmp13
    tmp23 = tmp22 + tmp16
    tmp24 = tmp23 - tmp13
    tmp25 = tmp18 * tmp24
    tl.store(out_ptr0 + (x3 + 2880 * y4), tmp25, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_18(in_ptr0, in_ptr1, out_ptr0,
    ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1000
    y1 = yindex // 1000
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1000 * x2 + 16000 * y1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_19(in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr):
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1000
    y1 = yindex // 1000
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1000 * x2 + 16000 * y1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp4, xmask)
    tl.store(out_ptr1 + (y0 + 1000 * x2 + 16000 * y1), tmp6, xmask)


@triton.jit
def triton_poi_fused_add_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_out_ptr0 + x0, xmask)
    tmp2 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32,
        primals_33, primals_34, primals_35, primals_36, primals_37,
        primals_38, primals_39, primals_40, primals_41, primals_42,
        primals_43, primals_44, primals_45, primals_46, primals_47,
        primals_48, primals_49, primals_50, primals_51, primals_52,
        primals_53, primals_54, primals_55, primals_56, primals_57,
        primals_58, primals_59, primals_60, primals_61, primals_62,
        primals_63, primals_64, primals_65, primals_66, primals_67,
        primals_68, primals_69, primals_70, primals_71, primals_72,
        primals_73, primals_74, primals_75, primals_76, primals_77,
        primals_78, primals_79, primals_80, primals_81, primals_82,
        primals_83, primals_84, primals_85, primals_86, primals_87,
        primals_88, primals_89, primals_90, primals_91, primals_92,
        primals_93, primals_94, primals_95, primals_96, primals_97,
        primals_98, primals_99, primals_100, primals_101, primals_102,
        primals_103, primals_104, primals_105, primals_106, primals_107,
        primals_108, primals_109, primals_110, primals_111, primals_112,
        primals_113, primals_114, primals_115, primals_116, primals_117,
        primals_118, primals_119, primals_120, primals_121, primals_122,
        primals_123, primals_124, primals_125, primals_126, primals_127,
        primals_128, primals_129, primals_130, primals_131, primals_132,
        primals_133, primals_134, primals_135, primals_136, primals_137,
        primals_138, primals_139, primals_140, primals_141, primals_142,
        primals_143, primals_144, primals_145, primals_146, primals_147,
        primals_148, primals_149, primals_150, primals_151, primals_152,
        primals_153, primals_154, primals_155, primals_156, primals_157,
        primals_158, primals_159, primals_160, primals_161, primals_162,
        primals_163, primals_164, primals_165, primals_166, primals_167,
        primals_168, primals_169, primals_17