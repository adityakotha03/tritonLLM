import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 131072 * x1), xmask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 131072 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (65536 + 2 * x0 + 131072 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65537 + 2 * x0 + 131072 * x1), xmask,
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
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 524288 * x1), xmask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 524288 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (262144 + 2 * x0 + 524288 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (262145 + 2 * x0 + 524288 * x1), xmask,
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
def triton_poi_fused__softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4096 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (1024 + x0 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2048 + x0 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3072 + x0 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4096 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (1024 + x0 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2048 + x0 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3072 + x0 + 4096 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 1048576 * x1), xmask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 1048576 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (524288 + 2 * x0 + 1048576 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (524289 + 2 * x0 + 1048576 * x1), xmask,
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
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 2048 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (512 + x0 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1024 + x0 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1536 + x0 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 2048 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (512 + x0 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1024 + x0 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1536 + x0 + 2048 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 2097152 * x1), xmask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 2097152 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1048576 + 2 * x0 + 2097152 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1048577 + 2 * x0 + 2097152 * x1), xmask,
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
def triton_poi_fused__softmax_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_14(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 4194304 * x1), xmask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 4194304 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2097152 + 2 * x0 + 4194304 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2097153 + 2 * x0 + 4194304 * x1), xmask,
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
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18,
    in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25,
    in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 128 % 128
    x0 = xindex % 128
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 128 * x1 + 8192 * x2), tmp4 & xmask, other
        =0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 128 * x1 + 8192 * x2), tmp4 & xmask, other
        =0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.load(in_ptr4 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr5 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.load(in_ptr6 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr7 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr8 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr9 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr10 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr11 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr12 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.load(in_ptr13 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.load(in_ptr14 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr15 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp35 = tmp33 * tmp34
    tmp36 = tl.load(in_ptr16 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp37 = tmp35 + tmp36
    tmp38 = tl.load(in_ptr17 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr18 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr19 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp43 = tmp41 * tmp42
    tmp44 = tl.load(in_ptr20 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr21 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp47 = tmp45 * tmp46
    tmp48 = tl.load(in_ptr22 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.load(in_ptr23 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp51 = tmp49 * tmp50
    tmp52 = tl.load(in_ptr24 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.load(in_ptr25 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp55 = tmp53 * tmp54
    tmp56 = tl.load(in_ptr26 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp57 = tmp55 + tmp56
    tmp58 = tl.load(in_ptr27 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp59 = tmp57 * tmp58
    tmp60 = tl.load(in_ptr28 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.load(in_ptr29 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp63 = tmp61 * tmp62
    tmp64 = tl.load(in_ptr30 + x1