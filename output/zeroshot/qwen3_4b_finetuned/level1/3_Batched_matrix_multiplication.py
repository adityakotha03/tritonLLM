import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tl.store(out_ptr0 + x3, tmp30, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13
    tmp12 = tmp7 - tmp8
    tmp14 = tmp9 - tmp7
    tmp16 = tmp14 - tmp12
    tmp17 = tl.where(xmask, tmp16, 0.0)
    tmp18 = tl.where(xmask, tmp14, 0.0)
    tmp19 = tl.where(xmask, tmp12, 0.0)
    tmp20 = tl.where(xmask, tmp10, 0.0)
    tmp21 = tl.where(xmask, tmp1, 0.0)
    tmp22 = tl.where(xmask, tmp3, 0.0)
    tmp23 = tl.where(xmask, tmp5, 0.0)
    tmp24 = tl.where(xmask, tmp17, tmp20)
    tmp25 = tl.where(xmask, tmp18, tmp21)
    tmp26 = tl.where(xmask, tmp19, tmp22)
    tmp27 = tl.where(xmask, tmp16, tmp23)
    tmp28 = tmp24 + tmp25
    tmp29 = tmp26 + tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = tmp0 / tmp30
    tmp32 = tmp1 / tmp30
    tmp33 = tmp3 / tmp30
    tmp34 = tmp5 / tmp30
    tmp35 = tl.where(xmask, tmp31, tmp32)
    tmp36 = tl.where(xmask, tmp33, tmp34)
    tl.store(out_ptr0 + x3, tmp35, xmask)
    tl.store(out_ptr0 + (256 + x3), tmp36, xmask)


@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp13 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp15 = tl.load(in_ptr0 + (768 + x0 + 1024 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp2
    tmp6 = tmp5 - tmp4
    tmp7 = tmp10 - tmp11
    tmp8 = tmp13 - tmp10
    tmp9 = tmp15 - tmp13