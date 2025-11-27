import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1280
    x1 = xindex // 1280
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1024 * x1 + x0), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 1280, tl.int64)
    tmp9 = tl.load(in_ptr1 + (128 * x1 + (-1024 + x0)), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (-1024 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(out_ptr0 + x2, tmp18, xmask)


@triton.jit
def triton_poi_fused_stack_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 < tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp13, tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp5)
    tl.store(out_ptr0 + x3, tmp19, xmask)


@triton.jit
def triton_poi_fused_stack_15(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 * x1 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0