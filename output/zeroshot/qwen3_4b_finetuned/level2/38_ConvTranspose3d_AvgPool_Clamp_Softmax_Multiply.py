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


@triton.jit
def triton_per_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + tl.broadcast_to(r0, [RBLOCK]), tmp2, None)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_1(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_2(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_3(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_convolution_4(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp8
    tmp9 = tmp7 + tmp10
    tmp11 = tmp9 + tmp13
    tl.store(out_ptr0 + x3, tmp11, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_5(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_6(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_7(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp8
    tmp9 = tmp7 + tmp10
    tmp11 = tmp9 + tmp13
    tmp12 = tmp11 + tmp15
    tl.store(out_ptr0 + x3, tmp12, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_9(out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_10(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_11(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp8
    tmp9 = tmp7 + tmp10
    tmp11 = tmp9 + tmp13
    tmp12 = tmp11 + tmp15
    tl.store(out_ptr0 + x3, tmp12, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_13(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_14(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_15(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp8
    tmp9 = tmp7 + tmp10
    tmp11 = tmp9 + tmp13
    tmp12 = tmp11 + tmp15
    tl.store(out_ptr0 + x3, tmp12, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_17(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_18(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_19(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp8
    tmp9 = tmp7 + tmp10
    tmp11 = tmp9 + tmp13
    tmp12 = tmp11 + tmp15
    tl.store(out_ptr0 + x3, tmp12, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_21(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_22(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_23(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp8
    tmp9 = tmp7 + tmp10
    tmp11 = tmp9 + tmp13
    tmp12 = tmp11 + tmp15
    tl.store(out_ptr0 + x3, tmp12, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_25(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_26(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_27(out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int