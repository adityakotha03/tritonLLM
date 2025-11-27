import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_cumsum_exp_0(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 16 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (3 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (4 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (5 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (6 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr0 + (7 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (8 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr0 + (9 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (10 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (11 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (12 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (13 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (14 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (15 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (16 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (17 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (18 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (19 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (20 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (21 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (22 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (23 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (24 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (25 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (26 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (27 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (28 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (29 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (30 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (31 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (32 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (33 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (34 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (35 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (36 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (37 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (38 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (39 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (40 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (41 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (42 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (43 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (44 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (45 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (46 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (47 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (48 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (49 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (50 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (51 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (52 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (53 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (54 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (55 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (56 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (57 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (58 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (59 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (60 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (61 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (62 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (63 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (64 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (65 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (66 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (67 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (68 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (69 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (70 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (71 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (72 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (73 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (74 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (75 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (76 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (77 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (78 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (79 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (80 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (81 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (82 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (83 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (84 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (85 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (86 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (87 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (88 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (89 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (90 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (91 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (92 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (93 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (94 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (95 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (96 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (97 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (98 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (99 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (100 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (101 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (102 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (103 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (104 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (105 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (106 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (107 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (108 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (109 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (110 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (111 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (112 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (113 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (114 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (115 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (116 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (117 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (118 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (119 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (120 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (121 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (122 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (123 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (124 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (125 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (126 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (127 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (128 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (129 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (130 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (131 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (132 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (133 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (134 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (135 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (136 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (137 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (138 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (139 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (140 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (141 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (142 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (143 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (144 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (145 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (146 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (147 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (148 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (149 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (150 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (151 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (152 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (153 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (154 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (155 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (156 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (157 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (158 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (159 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (160 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (161 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (162 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (163 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (164 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (165 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (166 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (167 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (168 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (169 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (170 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (171 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (172 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (173 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (174 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (175 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (176 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (177 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (178 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (179 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (180 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (181 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (182 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (183 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (184 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (185 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (186 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (187 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (188 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (189 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (190 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (191 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (192 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (193 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (194 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (195 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (196 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (197 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (198 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (199 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (200 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (201 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (202 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (203 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (204 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (205 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (206 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (207 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (208 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (209 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (210 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (211 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (212 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (213 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (214 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (215 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (216 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (217 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (218 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (219 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (220 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (221 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (222 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (223 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (224 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (225 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (226 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (227 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp229