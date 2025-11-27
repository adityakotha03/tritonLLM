import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_cumsum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (3 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (4 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (5 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (6 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (7 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (8 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (9 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (10 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (11 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (12 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (13 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (14 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (15 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (16 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (17 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (18 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (19 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (20 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (21 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (22 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (23 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (24 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (25 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (26 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (27 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (28 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (29 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (30 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (31 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (32 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (33 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (34 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (35 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (36 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (37 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (38 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (39 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (40 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (41 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (42 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (43 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (44 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (45 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (46 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (47 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (48 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (49 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (50 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (51 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (52 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (53 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (54 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (55 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (56 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (57 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (58 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (59 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (60 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (61 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (62 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (63 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (64 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (65 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (66 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (67 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (68 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (69 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (70 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (71 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (72 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (73 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (74 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (75 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (76 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (77 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (78 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (79 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (80 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (81 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (82 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (83 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (84 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (85 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (86 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (87 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (88 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (89 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (90 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (91 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (92 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (93 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (94 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (95 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (96 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (97 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (98 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (99 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (100 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (101 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (102 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (103 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (104 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (105 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (106 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (107 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (108 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (109 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (110 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (111 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (112 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (113 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (114 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (115 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (116 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (117 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (118 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (119 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (120 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (121 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (122 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (123 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (124 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (125 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (126 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (127 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (128 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (129 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (130 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (131 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (132 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (133 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (134 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (135 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (136 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (137 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (138 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (139 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (140 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (141 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (142 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (143 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (144 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (145 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (146 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (147 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (148 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (149 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (150 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (151 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (152 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (153 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (154 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (155 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (156 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (157 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (158 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (159 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (160 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (161 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (162 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (163 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (164 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (165 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (166 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (167 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (168 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (169 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (170 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (171 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (172 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (173 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (174 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (175 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (176 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (177 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (178 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (179 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (180 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (181 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (182 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (183 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (184 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (185 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (186 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (187 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (188 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (189 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (190 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (191 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (192 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (193 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (194 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (195 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (196 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (197 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (198 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (199 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (200 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (201 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (202 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (203 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (204 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (205 + x0 // 32768), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (206 + x0 //