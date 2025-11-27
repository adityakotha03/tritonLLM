import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp0 * tmp5
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 - tmp4
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = tmp7 * tmp8
    tmp12 = tmp3 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp7 + tmp13
    tl.store(out_ptr0 + x0, tmp14, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3072 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (4096 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5120 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (6144 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (7168 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (8192 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (9216 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (10240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (11264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr0 + (12288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (13312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (14336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (15360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (16384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (17408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (18432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (19456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (20480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (21504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr0 + (22528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (23552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (24576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (25600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp26 = tl.load(in_ptr0 + (26624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (27648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr0 + (28672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (29696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr0 + (30720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (31744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp32 = tl.load(in_ptr0 + (32768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (33792 + x0), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr0 + (34816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (35840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr0 + (36864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (37888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr0 + (38912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (39936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr0 + (40960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (41984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (43008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (44032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp44 = tl.load(in_ptr0 + (45056 + x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (46080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr0 + (47104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (48128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (49152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (50176 + x0), xmask, eviction_policy='evict_last'
        )
    tmp50 = tl.load(in_ptr0 + (51200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (52224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr0 + (53248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (54272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (55296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (56320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr0 + (57344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (58368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr0 + (59392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (60416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (61440 + x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (62464 + x0), xmask, eviction_policy='evict_last'
        )
    tmp62 = tl.load(in_ptr0 + (63488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (64512 + x0), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr0 + (65536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (66560 + x0), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (67584 + x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (68608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp68 = tl.load(in_ptr0 + (69632 + x0), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (70656 + x0), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr0 + (71680 + x0), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (72704 + x0), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (73728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (74752 + x0), xmask, eviction_policy='evict_last'
        )
    tmp74 = tl.load(in_ptr0 + (75776 + x0), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (76800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr0 + (77824 + x0), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (78848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (79872 + x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (80896 + x0), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr0 + (81920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (82944 + x0), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr0 + (83968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (84992 + x0), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (86016 + x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (87040 + x0), xmask, eviction_policy='evict_last'
        )
    tmp86 = tl.load(in_ptr0 + (88064 + x0), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (89088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr0 + (90112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (91136 + x0), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (92160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (93184 + x0), xmask, eviction_policy='evict_last'
        )
    tmp92 = tl.load(in_ptr0 + (94208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (95232 + x0), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr0 + (96256 + x0), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (97280 + x0), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (98304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (99328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp98 = tl.load(in_ptr0 + (100352 + x0), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (101376 + x0), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr0 + (102400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (103424 + x0), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (104448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (105472 + x0), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr0 + (106496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (107520 + x0), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr0 + (108544 + x0), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (109568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (110592 + x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (111616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp110 = tl.load(in_ptr0 + (112640 + x0), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (113664 + x0), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr0 + (114688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (115712 + x0), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (116736 + x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (117760 + x0), xmask, eviction_policy='evict_last'
        )
    tmp116 = tl.load(in_ptr0 + (118784 + x0), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (119808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr0 + (120832 + x0), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (121856 + x0), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (122880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (123904 + x0), xmask, eviction_policy='evict_last'
        )
    tmp122 = tl.load(in_ptr0 + (124928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (125952 + x0), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr0 + (126976 + x0), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr0 + (127999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (129024 + x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (130048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr0 + (131072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (132096 + x0), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr0 + (133120 + x0), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr0 + (134144 + x0), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (135168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (136192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp134 = tl.load(in_ptr0 + (137216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (138240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr0 + (139264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr0 + (140288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (141312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (142336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp140 = tl.load(in_ptr0 + (143360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (144384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr0 + (145408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr0 + (146432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (147456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (148480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp146 = tl.load(in_ptr0 + (149504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (150528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr0 + (151552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr0 + (152576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (153600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (154624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr0 + (155648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (156672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr0 + (157696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr0 + (158720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (159744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (160768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp158 = tl.load(in_ptr0 + (161792 + x0), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (162816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr0 + (163840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr0 + (164864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (165888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (166912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp164 = tl.load(in_ptr0 + (167936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (168960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr0 + (169984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr0 + (170000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (171024 + x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (172048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp170 = tl.load(in_ptr0 + (173072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (174096 + x0), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr0 + (175120 + x0), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr0 + (176144 + x0), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (177168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (178192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr0 + (179216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (180240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr0 + (181264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr0 + (182288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (183312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (184336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp182 = tl.load(in_ptr0 + (185360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (186384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr0 + (187408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr0 + (188432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (189456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (190480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp188 = tl.load(in_ptr0 + (191504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (192528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr0 + (193552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr0 + (194576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (195600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (196624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp194 = tl.load(in_ptr0 + (197648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (198672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr0 + (199696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr0 + (200720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (201744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (202768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr0 + (203792 + x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (204816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr0 + (205840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr0 + (206864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (207888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (208912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp206 = tl.load(in_ptr0 + (209936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (210960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr0 + (211984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr0 + (213008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (214032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (215056 + x0), xmask, eviction_policy='evict_last'
        )
    tmp212 = tl.load(in_ptr0 + (216080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (217104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl.load(in_ptr0 + (218128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp215 = tl.load(in_ptr0 + (219152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp216 = tl.load(in_ptr0 + (220176 + x0), xmask, eviction_policy='evict_last'
        )
    tmp217 = tl.load(in_ptr0 + (221200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp218 = tl.load(in_ptr0 + (222224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (223248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp220 = tl.load(in_ptr0 + (224272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp221 = tl.load(in_ptr0 + (225296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr0 + (226320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr0 + (227344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp224 = tl.load(in_ptr0 + (228368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (229392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp226 = tl.load(in_ptr0 + (230416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp227 = tl.load(in_ptr0 + (231440 + x0), xmask, eviction_policy='evict_last'
        )
    tmp228 = tl.load