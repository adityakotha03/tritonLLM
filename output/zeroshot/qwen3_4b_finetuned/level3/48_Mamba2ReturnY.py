import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_cumsum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 64 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (4 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr0 + (6 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (7 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (8 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (9 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (10 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (32 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (33 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (34 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (35 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (36 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (37 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (38 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (39 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (40 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (41 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (42 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (43 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (44 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (45 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (46 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (47 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (48 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (49 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (50 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (51 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (52 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (53 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (54 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (55 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (56 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (57 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (58 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (59 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (60 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (61 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (62 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (63 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (64 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (65 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (66 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (67 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (68 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (69 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (70 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (71 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (72 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (73 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (74 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (75 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (76 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (77 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (78 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (79 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (80 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (81 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (82 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (83 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (84 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (85 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (86 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (87 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (88 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (89 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (90 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (91 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (92 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (93 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (94 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (95 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (96 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (97 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (98 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (99 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (100 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (101 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (102 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (103 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (104 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (105 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (106 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (107 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (108 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (109 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (110 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (111 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (112 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (113 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (114 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (115 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (116 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (117 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (118 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (119 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (120 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (121 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (122 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (123 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (124 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (125 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (126 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (127 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (128 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (129 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (130 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (131 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (132 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (133 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (134 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (135 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_ptr0 + (136 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp274 = tl.load(in_ptr0 + (137 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (138 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_ptr0 + (139 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + (140 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (141 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp284 = tl.load(in_ptr0 + (142 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp286 = tl.load(in_ptr0 + (143 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (144 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + (145 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp292 = tl.load(in_ptr0 + (146 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (147 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (148 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp298 = tl.load(in_ptr0 + (149 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (150 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp302 = tl.load(in_ptr0 + (151 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp304 = tl.load(in_ptr0 + (152 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (153 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp308 = tl.load(in_ptr0 + (154 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + (155 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (156 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp314 = tl.load(in_ptr0 + (157 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp316 = tl.load(in_ptr0 + (158 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (159 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp320 = tl.load(in_ptr0 + (160 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp322 = tl.load(in_ptr0 + (161 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (162 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp326 = tl.load(in_ptr0 + (163 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp328 = tl.load(in_ptr0 + (164 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (165 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp332 = tl.load(in_ptr0 + (166 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp334 = tl.load(in_ptr0 + (167 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (168 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp338 = tl.load(in_ptr0 + (169 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp340 = tl.load(in_ptr0 + (170 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (171 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp344 = tl.load(in_ptr0 + (172 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp346 = tl.load(in_ptr0 + (173 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (174 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp350 = tl.load(in_ptr0 + (175 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp352 = tl.load(in_ptr0 + (176 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (177 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp356 = tl.load(in_ptr0 + (178 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp358 = tl.load(in_ptr0 + (179 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (180 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp362 = tl.load(in_ptr0 + (181 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp364 = tl.load(in_ptr0 + (182 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (183 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp368 = tl.load(in_ptr0 + (184 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp370 = tl.load(in_ptr0 + (185 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (186 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp374 = tl.load(in_ptr0 + (187 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp376 = tl.load(in_ptr0 + (188 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (189 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp380 = tl.load(in_ptr0 + (190 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp382 = tl.load(in_ptr0 + (191 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (192 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp386 = tl.load(in_ptr0 + (193 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp388 = tl.load(in_ptr0 + (194 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (195 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp392 = tl.load(in_ptr0 + (196 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp394 = tl.load(in_ptr0 + (197 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (198 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp398 = tl.load(in_ptr0 + (199 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp400 = tl.load(in_ptr0 + (200 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (201 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp404 = tl.load(in_ptr0 + (202 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp406 = tl.load(in_ptr0 + (203 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (204 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp410 = tl.load(in_ptr0 + (205 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp412 = tl.load(in_ptr0 + (206 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp414 = tl.load(in_ptr0 + (207 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp416 = tl.load(in_ptr0 + (208 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp418 = tl.load(in_ptr0 + (209 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp420 = tl.load(in_ptr0 + (210 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp422 = tl.load(in_ptr0 + (211 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp424 = tl.load(in_ptr0 + (212 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp426 = tl.load(in_ptr0 + (213 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp428 = tl.load(in_ptr0 + (214 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp430 = tl.load(in_ptr0 + (215 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp432 = tl.load(in_ptr0 + (216 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp434 = tl.load(in_ptr0 + (217 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp436 = tl.load(in_ptr0 + (218 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp438 = tl.load(in_ptr0 + (219 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp440 = tl.load(in_ptr0 + (220 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp442 = tl.load(in_ptr0 + (221 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp444 = tl.load(in_ptr0 + (222 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp446 = tl.load(in_ptr0 + (223 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp448 = tl.load(in_ptr0 + (224 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp450 = tl.load(in_ptr0 + (225 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp452 = tl.load(in_ptr