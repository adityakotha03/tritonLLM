import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_div_max_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tl.store(out_ptr0 + x4, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_1(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x4, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_2(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tmp6 & tmp10
    tmp20 = tmp14 & tmp18
    tmp21 = tmp19 & tmp20
    tmp22 = tmp21 & tmp13
    tmp23 = tmp22 & tmp16
    tl.store(out_ptr0 + x4, tmp23, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_3(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tmp6 & tmp10
    tmp20 = tmp14 & tmp18
    tmp21 = tmp19 & tmp20
    tmp22 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp23 = tmp22 + tmp1
    tmp24 = tmp23 * tmp3
    tmp25 = tmp24 > tmp5
    tmp26 = tmp21 & tmp25
    tmp27 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp1
    tmp29 = tmp28 * tmp3
    tmp30 = tmp29 > tmp5
    tmp31 = tmp26 & tmp30
    tl.store(out_ptr0 + x4, tmp31, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_4(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 > tmp5
    tmp23 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tmp18 & tmp22
    tmp28 = tmp26 & tmp27
    tmp29 = tl.load(in_ptr0 + (96 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp1
    tmp31 = tmp30 * tmp3
    tmp32 = tmp31 > tmp5
    tmp33 = tmp28 & tmp32
    tl.store(out_ptr0 + x4, tmp33, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_5(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 > tmp5
    tmp23 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tl.load(in_ptr0 + (96 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp1
    tmp29 = tmp28 * tmp3
    tmp30 = tmp29 > tmp5
    tmp31 = tmp22 & tmp26
    tmp32 = tmp30 & tmp31
    tmp33 = tl.load(in_ptr0 + (112 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp34 = tmp33 + tmp1
    tmp35 = tmp34 * tmp3
    tmp36 = tmp35 > tmp5
    tmp37 = tmp32 & tmp36
    tl.store(out_ptr0 + x4, tmp37, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_6(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 > tmp5
    tmp23 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tl.load(in_ptr0 + (96 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp1
    tmp29 = tmp28 * tmp3
    tmp30 = tmp29 > tmp5
    tmp31 = tl.load(in_ptr0 + (112 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp32 = tmp31 + tmp1
    tmp33 = tmp32 * tmp3
    tmp34 = tmp33 > tmp5
    tmp35 = tmp26 & tmp30
    tmp36 = tmp34 & tmp35
    tmp37 = tl.load(in_ptr0 + (128 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp38 = tmp37 + tmp1
    tmp39 = tmp38 * tmp3
    tmp40 = tmp39 > tmp5
    tmp41 = tmp36 & tmp40
    tl.store(out_ptr0 + x4, tmp41, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_7(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 > tmp5
    tmp23 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tl.load(in_ptr0 + (96 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp1
    tmp29 = tmp28 * tmp3
    tmp30 = tmp29 > tmp5
    tmp31 = tl.load(in_ptr0 + (112 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp32 = tmp31 + tmp1
    tmp33 = tmp32 * tmp3
    tmp34 = tmp33 > tmp5
    tmp35 = tl.load(in_ptr0 + (128 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp36 = tmp35 + tmp1
    tmp37 = tmp36 * tmp3
    tmp38 = tmp37 > tmp5
    tmp39 = tmp30 & tmp34
    tmp40 = tmp38 & tmp39
    tmp41 = tl.load(in_ptr0 + (144 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp42 = tmp41 + tmp1
    tmp43 = tmp42 * tmp3
    tmp44 = tmp43 > tmp5
    tmp45 = tmp40 & tmp44
    tl.store(out_ptr0 + x4, tmp45, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_8(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 > tmp5
    tmp23 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tl.load(in_ptr0 + (96 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp1
    tmp29 = tmp28 * tmp3
    tmp30 = tmp29 > tmp5
    tmp31 = tl.load(in_ptr0 + (112 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp32 = tmp31 + tmp1
    tmp33 = tmp32 * tmp3
    tmp34 = tmp33 > tmp5
    tmp35 = tl.load(in_ptr0 + (128 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp36 = tmp35 + tmp1
    tmp37 = tmp36 * tmp3
    tmp38 = tmp37 > tmp5
    tmp39 = tl.load(in_ptr0 + (144 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp40 = tmp39 + tmp1
    tmp41 = tmp40 * tmp3
    tmp42 = tmp41 > tmp5
    tmp43 = tl.load(in_ptr0 + (160 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp44 = tmp43 + tmp1
    tmp45 = tmp44 * tmp3
    tmp46 = tmp45 > tmp5
    tmp47 = tmp34 & tmp42
    tmp48 = tmp46 & tmp47
    tmp49 = tl.load(in_ptr0 + (176 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp50 = tmp49 + tmp1
    tmp51 = tmp50 * tmp3
    tmp52 = tmp51 > tmp5
    tmp53 = tmp48 & tmp52
    tl.store(out_ptr0 + x4, tmp53, xmask)


@triton.jit
def triton_poi_fused_add_div_max_pool3d_9(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 198400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 64
    x2 = xindex // 1024 % 64
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 1024 * x1 + 65536 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 > tmp5
    tmp11 = tl.load(in_ptr0 + (32 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tmp11 + tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tl.load(in_ptr0 + (48 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp1
    tmp17 = tmp16 * tmp3
    tmp18 = tmp17 > tmp5
    tmp19 = tl.load(in_ptr0 + (64 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp1
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 > tmp5
    tmp23 = tl.load(in_ptr0 + (80 + x0 + 16 * x2 + 1024 * x1 + 65536 * x3),
        xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tl.load(in