import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_6(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_9(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_10(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp8 = tmp7 > tmp1
    tmp10 = tl.where(tmp8, tmp7, tmp1)
    tmp12 = tl.where(tmp6, tmp6, tmp10)
    tmp14 = tmp5 > tmp12
    tmp15 = tl.where(tmp14, tmp5, tmp12)
    tmp16 = tmp9 > tmp12
    tmp17 = tl.where(tmp16, tmp9, tmp12)
    tmp18 = tl.where(tmp15, tmp15, tmp17)
    tmp19 = tmp11 > tmp12
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tl.where(tmp18, tmp18, tmp20)
    tmp22 = tmp13 > tmp12
    tmp23 = tl.where(tmp22, tmp13, tmp12)
    tmp24 = tl.where(tmp21, tmp21, tmp23)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tmp24 == tmp25
    tl.store(out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_12(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0