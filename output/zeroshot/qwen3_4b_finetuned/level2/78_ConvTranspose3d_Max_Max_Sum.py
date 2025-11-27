import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_max_pool3d_with_indices_1(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (2 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (5 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (6 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (7 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (8 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (9 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (10 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (11 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (12 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (13 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (14 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (15 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (16 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (17 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (18 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (19 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (20 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (21 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (22 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (23 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (24 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (25 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (26 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (27 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (28 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (29 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (30 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr0 + (31 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (32 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (33 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (34 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr0 + (35 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr0 + (36 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (37 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr0 + (38 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (39 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr0 + (40 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr0 + (41 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (42 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr0 + (43 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr0 + (44 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr0 + (45 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr0 + (46 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr0 + (47 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (48 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr0 + (49 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (50 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr0 + (51 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr0 + (52 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr0 + (53 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr0 + (54 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (55 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr0 + (56 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr0 + (57 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (58 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr0 + (59 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr0 + (60 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr0 + (61 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr0 + (62 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (63 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp7 = tmp2 | tmp4
    tmp8 = tl.where(tmp7, tmp0, tmp1)
    tmp9 = tmp5 > tmp8
    tmp10 = tmp7 | tmp9
    tmp11 = tl.where(tmp10, tmp5, tmp8)
    tmp12 = tmp6 > tmp11
    tmp13 = tmp10 | tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp11)
    tmp15 = tmp7 & tmp13
    tmp16 = tl.where(tmp15, tmp8, tmp14)
    tmp17 = tmp12 & tmp13
    tmp18 = tl.where(tmp17, tmp11, tmp14)
    tmp19 = tl.where(tmp15, tmp16, tmp18)
    tmp20 = tmp19 + 0.0
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tmp21 < tmp20
    tmp23 = tl.load(in_ptr0 + (64 + 4 * x3), tmp22 & tmp15, eviction_policy=
        'evict_last', other=0.0)
    tmp24 = tmp23 > tmp16
    tmp25 = tmp22 & tmp17
    tmp26 = tl.where(tmp25, tmp23, tmp18)
    tmp27 = tmp24 | tmp25
    tmp28 = tl.where(tmp27, tmp23, tmp19)
    tmp29 = tmp28 + 0.0
    tmp30 = tl.full([1], 0, tl.int64)
    tmp31 = tmp30 < tmp29
    tmp32 = tl.load(in_ptr0 + (128 + 4 * x3), tmp31 & tmp27, eviction_policy
        ='evict_last', other=0.0)
    tmp33 = tmp32 > tmp28
    tmp34 = tmp31 & tmp27
    tmp35 = tl.where(tmp34, tmp32, tmp28)
    tmp36 = tmp33 | tmp34
    tmp37 = tl.where(tmp36, tmp32, tmp29)
    tmp38 = tmp37 + 0.0
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp39 < tmp38
    tmp41 = tl.load(in_ptr0 + (256 + 4 * x3), tmp40 & tmp36, eviction_policy
        ='evict_last', other=0.0)
    tmp42 = tmp41 > tmp37
    tmp43 = tmp40 & tmp36
    tmp44 = tl.where(tmp43, tmp41, tmp37)
    tmp45 = tmp42 | tmp43
    tmp46 = tl.where(tmp45, tmp41, tmp38)
    tmp47 = tmp46 + 0.0
    tmp48 = tl.full([1], 0, tl.int64)
    tmp49 = tmp48 < tmp47
    tmp50 = tl.load(in_ptr0 + (512 + 4 * x3), tmp49 & tmp45, eviction_policy
        ='evict_last', other=0.0)
    tmp51 = tmp50 > tmp46
    tmp52 = tmp49 & tmp45
    tmp53 = tl.where(tmp52, tmp50, tmp46)
    tmp54 = tmp51 | tmp52
    tmp55 = tl.where(tmp54, tmp50, tmp47)
    tmp56 = tmp55 + 0.0
    tl.store(out_ptr0 + x3, tmp56, xmask)
    tl.store(out_ptr1 + (x0 + 64 * x1 + 4096 * x2), tmp56, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_2(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 32
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (5 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (6 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (7 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (8 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (9 + 4 * x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (10 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (11 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (12 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (13 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (14 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (15 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (16 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (17 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (18 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (19 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (20 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (21 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (22 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (23 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (24 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (25 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (26 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (27 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (28 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (29 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (30 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (31 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (32 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (33 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (34 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (35 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (36 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (37 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (38 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (39 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (40 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (41 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (42 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (43 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (44 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (45 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (46 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (47 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (48 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (49 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (50 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (51 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (52 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (53 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (54 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (55 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (56 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (57 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (58 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (59 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (60 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (61 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (62 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (63 + 4 * x3), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 > tmp1
    tmp6 = tl.where(tmp4, tmp3, tmp1)
    tmp7 = tmp2 | tmp4
    tmp8 = tl.where(tmp7, tmp0, tmp1)
    tmp9 = tmp5 > tmp8
    tmp10 = tmp7 | tmp9
    tmp11 = tl.where(tmp10, tmp5, tmp8)
    tmp12 = tmp6 > tmp11
    tmp13 = tmp10 | tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp11)
    tmp15 = tmp7 & tmp13
    tmp16 = tl.where(tmp15, tmp8, tmp14)
    tmp17 = tmp12 & tmp13
    tmp18 = tl.where(tmp17, tmp11, tmp14)
    tmp19 = tl.where(tmp15, tmp16, tmp18)
    tmp20 = tmp19 + 0.0
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tmp21 < tmp20
    tmp23 = tl.load(in_ptr0 + (64 + 4 * x3), tmp22 & tmp15, eviction_policy
        ='evict_last', other=0.0)
    tmp24 = tmp23 > tmp16
    tmp25 = tmp22 & tmp17
    tmp26 = tl.where(tmp25, tmp23, tmp18)
    tmp27 = tmp24 | tmp25
    tmp28 = tl.where(tmp27, tmp23, tmp19)
    tmp29 = tmp28 + 0.0
    tmp30 = tl.full([1], 0, tl.int64)
    tmp31 = tmp30 < tmp29
    tmp32 = tl.load(in_ptr0 + (128 + 4 * x3), tmp31 & tmp27, eviction_policy
        ='evict_last', other=0.0)
    tmp33 = tmp32 > tmp28
    tmp34 = tmp31 & tmp27
    tmp35 = tl.where(tmp34, tmp32, tmp28)
    tmp36 = tmp33 | tmp34
    tmp37 = tl.where(tmp36, tmp32, tmp29)
    tmp38 = tmp37 + 0.0
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp39 < tmp38
    tmp41 = tl.load(in_ptr0 + (256 + 4 * x3), tmp40 & tmp36, eviction_policy
        ='evict_last', other=0.0)
    tmp42 = tmp41 > tmp37
    tmp43 = tmp40 & tmp36
    tmp44 = tl.where(tmp43, tmp41, tmp37)
    tmp45 = tmp42 | tmp43
    tmp46 = tl.where(tmp45, tmp41, tmp38)
    tmp47 = tmp46 + 0.0
    tmp48 = tl.full([1], 0, tl.int64)
    tmp49 = tmp48 < tmp47
    tmp50 = tl.load(in_ptr0 + (512 + 4 * x3), tmp49 & tmp45, eviction_policy
        ='evict_last', other=0.0)
    tmp51 = tmp50 > tmp46
    tmp52 = tmp49 & tmp45
    tmp53 = tl.where(tmp52, tmp50, tmp46)
    tmp54 = tmp51 | tmp52
    tmp55 = tl.where(tmp54, tmp50, tmp47)
    tmp56 = tmp55 + 0.0
    tl.store(out_ptr0 + x3, tmp56, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 32, 32, 32), (524288, 16384, 512, 16,
        1))
    assert_size_stride(arg1_1, (64, 32, 5, 5, 5), (40960, 1280, 256, 51, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 32, 33, 33, 33), (3596448, 112500, 3375,
            101, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_max_pool3d_with_indices_1[grid(4608)](arg1_1,
            buf0, buf0, 4608, XBLOCK=128, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((16, 32, 33, 33, 33), (3596448, 112500, 3375,
            101, 1), torch.float32)
        triton_poi_fused_max_pool3d_with_indices_2[grid(1536)](buf0, buf1, 
            1536, XBLOCK=128, num_warps=8, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((16, 1, 1, 1, 1), (1, 64, 64, 64, 64), torch
            .float32)
        triton_poi_fused_sum_0[grid(16)](buf1, buf2, 16, XBLOCK=16,
            num_warps=1, num_stages=1)
        del buf1
    return buf2, arg0_1


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, input_0):
        arg1_1 = self.conv_transpose.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
