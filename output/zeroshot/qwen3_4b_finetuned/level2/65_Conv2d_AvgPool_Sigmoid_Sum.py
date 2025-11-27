import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 516064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = xindex // 384 % 384
    x2 = xindex // 153312
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (384 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (385 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (384 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (385 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (384 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (385 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (768 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (769 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (1536 + 2 * x0 + 768 * x1 + 307200 * x2), xmask
        )
    tmp23 = tl.load(in_ptr0 + (1537 + 2 * x0 + 768 * x1 + 307200 * x2), xmask
        )
    tmp26 = tl.load(in_ptr0 + (768 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (769 + 2 * x0 + 768 * x1 + 307200 * x2), xmask,
        eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (1536 + 2 * x0 + 768 * x1 + 307200 * x2), xmask
        )
    tmp31 = tl.load(in_ptr0 + (1537 + 2 * x0 + 768 * x1 + 307200 * x2), xmask
        )
    tmp2 = tmp1 + tmp3
    tmp4 = tmp2 + tmp5
    tmp6 = tmp7 + tmp8
    tmp9 = tmp6 + tmp10
    tmp11 = tmp9 + tmp12
    tmp13 = tmp11 + tmp15
    tmp14 = tmp13 + tmp16
    tmp17 = tmp14 + tmp19
    tmp18 = tmp17 + tmp20
    tmp21 = tmp18 + tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24 + tmp26
    tmp28 = tmp25 + tmp29
    tmp30 = tmp28 + tmp31
    tmp32 = 0.0625
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + x3, tmp33, xmask)


@triton.jit
def triton_poi_fused_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5120 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6144 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7168 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + x0, tmp14, xmask)


@triton.jit
def triton_poi_fused_sum_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5120 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6144 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7168 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8192 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9216 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10240 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11264 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12288 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13312 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr0 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_sum_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5120 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6144 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7168 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8192 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9216 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10240 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11264 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12288 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13312 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14336 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15360 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16384 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17408 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18432 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19456 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20480 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21504 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22528 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23552 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24576 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25600 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26624 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27648 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28672 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29696 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30720 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31744 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32768 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33792 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34816 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35840 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36864 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37888 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38912 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39936 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40960 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41984 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (43008 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (44032 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (45056 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (46080 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (47104 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (48128 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (49152 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (50176 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (51200 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (52224 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp103 = tl.load(in_ptr0 + (53248 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp105 = tl.load(in_ptr0 + (54272 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp107 = tl.load(in_ptr0 + (55296 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp109 = tl.load(in_ptr0 + (56320 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp111 = tl.load(in_ptr0 + (57344 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp113 = tl.load(in_ptr0 + (58368 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp115 = tl.load(in_ptr0 + (59392 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp117 = tl.load(in_ptr0 + (60416 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp119 = tl.load(in_ptr0 + (61440 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp121 = tl.load(in_ptr0 + (62464 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp123 = tl.load(in_ptr0 + (63488 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp125 = tl.load(in_ptr0 + (64512 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp127 = tl.load(in_ptr0 + (65536 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp129 = tl.load(in_ptr0 + (66560 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp131 = tl.load(in_ptr0 + (67584 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp133 = tl.load(in_ptr0 + (68608 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp135 = tl.load(in_ptr0 + (69632 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp137 = tl.load(in_ptr0 + (70656 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp139 = tl.load(in_ptr0 + (71680 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp141 = tl.load(in_ptr0 + (72704 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp143 = tl.load(in_ptr0 + (73728 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp145 = tl.load(in_ptr0 + (74752 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp147 = tl.load(in_ptr0 + (75776 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp149 = tl.load(in_ptr0 + (76800 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp151 = tl.load(in_ptr0 + (77824 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp153 = tl.load(in_ptr0 + (78848 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp155 = tl.load(in_ptr0 + (79872 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp157 = tl.load(in_ptr0 + (80896 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp159 = tl.load(in_ptr0 + (81920 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp161 = tl.load(in_ptr0 + (82944 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp163 = tl.load(in_ptr0 + (83968 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp165 = tl.load(in_ptr0 + (84992 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp167 = tl.load(in_ptr0 + (85024 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp169 = tl.load(in_ptr0 + (86048 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp171 = tl.load(in_ptr0 + (87072 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp173 = tl.load(in_ptr0 + (88096 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp175 = tl.load(in_ptr0 + (89120 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp177 = tl.load(in_ptr0 + (90144 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp179 = tl.load(in_ptr0 + (91168 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp181 = tl.load(in_ptr0 + (92192 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp183 = tl.load(in_ptr0 + (93216 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp185 = tl.load(in_ptr0 + (94240 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp187 = tl.load(in_ptr0 + (95264 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp189 = tl.load(in_ptr0 + (96288 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp191 = tl.load(in_ptr0 + (97312 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp193 = tl.load(in_ptr0 + (98336 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp195 = tl.load(in_ptr0 + (99360 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp197 = tl.load(in_ptr0 + (100384 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp199 = tl.load(in_ptr0 + (101408 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp201 = tl.load(in_ptr0 + (102432 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp203 = tl.load(in_ptr0 + (103456 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp205 = tl.load(in_ptr0 + (104480 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp207 = tl.load(in_ptr0 + (105504 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp209 = tl.load(in_ptr0 + (106528 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp211 = tl.load(in_ptr0 + (107552 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp213 = tl.load(in_ptr0 + (108576 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp215 = tl.load(in_ptr0 + (109600 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp217 = tl.load(in_ptr0 + (110624 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp219 = tl.load(in_ptr0 + (111648 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp221 = tl.load(in_ptr0 + (112672 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp223 = tl.load(in_ptr0 + (113696 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp225 = tl.load(in_ptr0 + (114720 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp227 = tl.load(in_ptr0 + (115744 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp229 = tl.load(in_ptr0 + (116768 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp231 = tl.load(in_ptr0 + (117792 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp233 = tl.load(in_ptr0 + (118816 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp235 = tl.load(in_ptr0 + (119840 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp237 = tl.load(in_ptr0 + (120864 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp239 = tl.load(in_ptr0 + (121888 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp241 = tl.load(in_ptr0 + (122912 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp243 = tl.load(in_ptr0 + (123936 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp245 = tl.load(in_ptr0 + (124960 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp247 = tl.load(in_ptr0 + (125984 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp249 = tl.load(in_ptr0 + (126016 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp251 = tl.load(in_ptr0 + (127040 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp253 = tl.load(in_ptr0 + (128064 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp255 = tl.load(in_ptr0 + (129088 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp257 = tl.load(in_ptr0 + (130112 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp259 = tl.load(in_ptr0 + (131136 + 4096 * x0), xmask, eviction_policy
        ='evict_last')
    tmp261 = tl.load(in_ptr0 + (132160 + 4