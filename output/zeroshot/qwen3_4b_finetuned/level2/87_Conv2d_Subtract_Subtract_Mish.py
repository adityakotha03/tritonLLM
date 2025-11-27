import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 1024 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1025 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2050 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3075 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4100 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5125 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6150 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7175 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8200 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9225 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (10240 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11265 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12300 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13325 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14350 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15375 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16400 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17425 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18450 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19475 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20500 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21525 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22550 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23575 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24600 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25625 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26650 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27675 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28700 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29725 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30750 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31775 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (32800 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (33825 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (34850 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (35875 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (36900 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (37925 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (38950 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (39975 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (41000 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (42025 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (43050 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (44075 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (45100 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (46125 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (47150 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (48175 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (49200 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (50225 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (51250 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (52275 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (53300 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (54325 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (55350 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (56375 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (57400 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (58425 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (59450 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (60475 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (61500 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (62525 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (63550 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (64575 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (65600 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (66625 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (67650 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (68675 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (69700 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (70725 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (71750 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (72775 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (73800 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (74825 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (75850 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (76875 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (77900 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (78925 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (79950 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (80975 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (81999 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (83024 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (84049 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (85074 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (86099 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (87124 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (88149 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (89174 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (90199 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (91224 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (92249 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (93274 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (94299 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (95324 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (96349 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (97374 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (98399 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (99424 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (100449 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (101474 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (102500 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (103525 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (104550 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (105575 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (106600 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (107625 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (108650 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (109675 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (110700 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (111725 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (112750 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (113775 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (114800 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (115825 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (116850 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (117875 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (118900 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (119925 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (120950 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (121975 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (122999 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (124024 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (125049 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (126074 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (127099 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (128124 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (129149 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (130174 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (131199 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (132224 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (133249 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (134274 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (135299 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (136324 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (137349 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (138374 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_ptr0 + (139399 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp274 = tl.load(in_ptr0 + (140424 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (141449 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_ptr0 + (142474 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + (143499 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (144524 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp284 = tl.load(in_ptr0 + (145549 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp286 = tl.load(in_ptr0 + (146574 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (147599 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + (148624 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp292 = tl.load(in_ptr0 + (149649 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (150674 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (151699 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp298 = tl.load(in_ptr0 + (152724 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (153749 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp302 = tl.load(in_ptr0 + (154774 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp304 = tl.load(in_ptr0 + (155799 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (156824 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp308 = tl.load(in_ptr0 + (157849 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + (158874 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (159899 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp314 = tl.load(in_ptr0 + (160924 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp316 = tl.load(in_ptr0 + (161949 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (162974 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp320 = tl.load(in_ptr0 + (163999 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp322 = tl.load(in_ptr0 + (165024 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (166049 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp326 = tl.load(in_ptr0 + (167074 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp328 = tl.load(in_ptr0 + (168099 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (169124 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp332 = tl.load(in_ptr0 + (170149 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp334 = tl.load(in_ptr0 + (171174 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (172199 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp338 = tl.load(in_ptr0 + (173224 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp340 = tl.load(in_ptr0 + (174249 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (175274 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp344 = tl.load(in_ptr0 + (176299 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp346 = tl.load(in_ptr0 + (177324 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (178349 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp350 = tl.load(in_ptr0 + (179374 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp352 = tl.load(in_ptr0 + (180399 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (181424 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp356 = tl.load(in_ptr0 + (182449 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp358 = tl.load(in_ptr0 + (183474 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (184499 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp362 = tl.load(in_ptr0 + (185524 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp364 = tl.load(in_ptr0 + (186549 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (187574 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp368 = tl.load(in_ptr0 + (188599 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp370 = tl.load(in_ptr0 + (189624 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (190649 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp374 = tl.load(in_ptr0 + (191674 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp376 = tl.load(in_ptr0 + (192699 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (193724 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp380 = tl.load(in_ptr0 + (194749 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp382 = tl.load(in_ptr0 + (195774 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (196799 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp386 = tl.load(in_ptr0 + (197824 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp388 = tl.load(in_ptr0 + (198849 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (199874 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp392 = tl.load(in_ptr0 + (200899 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp394 = tl.load(in_ptr0 + (201924 + 1024 * x0), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (202949 + 1024 * x0), xmask, eviction_policy=
        'evict