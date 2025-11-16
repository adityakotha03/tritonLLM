import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 65535 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused_div_mean_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 65535
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 65535 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (4 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (8 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (9 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (10 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (32 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (33 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (34 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (35 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (36 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (37 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (38 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (39 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (40 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (41 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (42 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (43 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (44 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (45 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (46 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (47 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (48 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (49 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (50 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (51 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (52 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (53 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (54 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (55 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (56 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (57 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (58 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (59 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (60 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (61 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (62 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (63 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (64 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (65 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (66 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (67 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (68 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (69 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (70 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (71 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (72 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (73 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (74 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (75 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (76 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (77 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (78 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (79 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (80 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (81 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (82 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (83 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (84 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (85 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (86 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (87 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (88 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (89 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (90 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (91 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (92 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (93 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (94 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (95 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (96 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (97 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (98 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (99 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (100 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (101 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (102 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (103 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (104 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (105 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (106 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (107 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (108 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (109 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (110 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (111 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (112 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (113 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (114 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (115 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (116 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (117 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (118 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (119 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (120 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (121 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (122 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (123 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (124 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (125 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (126 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (127 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (128 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (129 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (130 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (131 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (132 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (133 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (134 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (135 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_ptr0 + (136 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp274 = tl.load(in_ptr0 + (137 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (138 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_ptr0 + (139 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + (140 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (141 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp284 = tl.load(in_ptr0 + (142 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp286 = tl.load(in_ptr0 + (143 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (144 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + (145 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp292 = tl.load(in_ptr0 + (146 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (147 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (148 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp298 = tl.load(in_ptr0 + (149 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (150 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp302 = tl.load(in_ptr0 + (151 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp304 = tl.load(in_ptr0 + (152 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (153 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp308 = tl.load(in_ptr0 + (154 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + (155 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (156 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp314 = tl.load(in_ptr0 + (157 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp316 = tl.load(in_ptr0 + (158 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (159 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp320 = tl.load(in_ptr0 + (160 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp322 = tl.load(in_ptr0 + (161 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (162 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp326 = tl.load(in_ptr0 + (163 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp328 = tl.load(in_ptr0 + (164 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (165 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp332 = tl.load(in_ptr0 + (166 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp334 = tl.load(in_ptr0 + (167 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (168 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp338 = tl.load(in_ptr0 + (169 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp340 = tl.load(in_ptr0 + (170 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (171 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp344 = tl.load(in_ptr0 + (172 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp346 = tl.load(in_ptr0 + (173 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (174 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp350 = tl.load(in_ptr0 + (175 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp352 = tl.load(in_ptr0 + (176 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (177 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp356 = tl.load(in_ptr0 + (178 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp358 = tl.load(in_ptr0 + (179 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (180 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp362 = tl.load(in_ptr0 + (181 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp364 = tl.load(in_ptr0 + (182 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (183 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp368 = tl.load(in_ptr0 + (184 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp370 = tl.load(in_ptr0 + (185 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (186 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp374 = tl.load(in_ptr0 + (187 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp376 = tl.load(in_ptr0 + (188 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (189 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp380 = tl.load(in_ptr0 + (190 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp382 = tl.load(in_ptr0 + (191 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (192 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp386 = tl.load(in_ptr0 + (193 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp388 = tl.load(in_ptr0 + (194 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (195 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp392 = tl.load(in_ptr0 + (196 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp394 = tl.load(in_ptr0 + (197 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (198 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp398 = tl.load(in_ptr0 + (199 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp400 = tl.load(in_ptr0 + (200 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (201 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp404 = tl.load(in_ptr0 + (202 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp406 = tl.load(in_ptr0 + (203 + 65535 * x1), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (204 +