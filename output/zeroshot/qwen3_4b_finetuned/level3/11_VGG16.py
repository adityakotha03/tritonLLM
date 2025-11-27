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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 27 * y1), tmp0, xmask & ymask)


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
    tmp0 = tl.load(in_ptr0 + (x2 + 2016 * y3), ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 6048 * y1), tmp0, ymask)


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
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 128 * x2 + 1152 * y1), tmp0, xmask)


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
    y0 = yindex % 256
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 256 * x2 + 2304 * y1), tmp0, xmask)


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
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 512 * x2 + 4608 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + 784 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 3 * x2 + 2352 * y1), tmp4, xmask & ymask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12
    x1 = xindex // 12 % 12
    x2 = xindex // 144
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 48 * x1 + 1152 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 48 * x1 + 1152 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (24 + 2 * x0 + 48 * x1 + 1152 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (25 + 2 * x0 + 48 * x1 + 1152 * x2), xmask,
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
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_8(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 128 * x2 + 131072 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_9(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = xindex // 8 % 8
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 32 * x1 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 32 * x1 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2 * x0 + 32 * x1 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2 * x0 + 32 * x1 + 2048 * x2), xmask,
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
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_10(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 2304 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 256 * x2 + 589824 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = xindex // 4 % 4
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 16 * x1 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 16 * x1 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2 * x0 + 16 * x1 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2 * x0 + 16 * x1 + 4096 * x2), xmask,
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
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_12(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 3136 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 512 * x2 + 1612800 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_13(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_14(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 3136 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 512 * x2 + 1612800 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_15(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_16(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 3136 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 512 * x2 + 1612800 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_17(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_18(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 3136 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 512 * x2 + 1612800 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_19(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_20(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 3136 * y3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (y0 + 512 * x2 + 1612800 * y1), tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_21(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel