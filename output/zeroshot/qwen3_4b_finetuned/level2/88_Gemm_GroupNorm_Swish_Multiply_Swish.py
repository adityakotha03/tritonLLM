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
def triton_poi_fused_add_group_norm_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 256 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (42 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (43 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (44 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (45 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (46 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (47 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (48 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (49 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (50 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (51 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (52 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (53 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (54 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (55 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (56 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (57 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (58 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (59 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (60 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (61 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (62 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (63 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (64 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (65 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (66 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (67 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (68 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (69 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (70 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (71 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (72 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (73 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (74 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (75 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (76 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (77 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (78 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (79 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (80 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (81 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (82 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (83 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (84 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (85 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (86 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (87 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (88 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (89 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (90 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (91 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (92 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (93 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (94 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (95 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (96 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (97 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (98 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (99 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (100 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (101 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (102 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (103 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (104 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (105 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (106 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (107 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (108 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (109 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (110 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (111 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (112 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (113 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (114 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (115 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (116 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (117 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (118 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (119 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (120 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (121 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (122 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (123 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (124 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (125 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (126 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (127 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (128 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (129 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (130 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (131 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (132 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (133 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (134 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (135 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (136 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (137 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (138 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_ptr0 + (139 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (140 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp281 = tl.load(in_ptr0 + (141 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp283 = tl.load(in_ptr0 + (142 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (143 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (144 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (145 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (146 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp293 = tl.load(in_ptr0 + (147 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + (148 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (149 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (150 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp301 = tl.load(in_ptr0 + (151 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (152 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + (153 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp307 = tl.load(in_ptr0 + (154 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (155 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (156 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp313 = tl.load(in_ptr0 + (157 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (158 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (159 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp319 = tl.load(in_ptr0 + (160 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (161 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (162 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp325 = tl.load(in_ptr0 + (163 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (164 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp329 = tl.load(in_ptr0 + (165 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (166 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (167 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (168 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp337 = tl.load(in_ptr0 + (169 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (170 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp341 = tl.load(in_ptr0 + (171 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp343 = tl.load(in_ptr0 + (172 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (173 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (174 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp349 = tl.load(in_ptr0 + (175 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (176 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp353 = tl.load(in_ptr0 + (177 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp355 = tl.load(in_ptr0 + (178 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (179 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (180 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp361 = tl.load(in_ptr0 + (181 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (182 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp365 = tl.load(in_ptr0 + (183 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp367 = tl.load(in_ptr0 + (184 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (185 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp371 = tl.load(in_ptr0 + (186 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp373 = tl.load(in_ptr0 + (187 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (188 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp377 = tl.load(in_ptr0 + (189 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp379 = tl.load(in_ptr0 + (190 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (191 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp383 = tl.load(in_ptr0 + (192 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp385 = tl.load(in_ptr0 + (193 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (194 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp389 = tl.load(in_ptr0 + (195 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp391 = tl.load(in_ptr0 + (196 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (197 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp395 = tl.load(in_ptr0 + (198 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp397 = tl.load(in_ptr0 + (199 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (200 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp401 = tl.load(in_ptr0 + (201 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp403 = tl.load(in_ptr0 + (202 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (203 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp407 = tl.load(in_ptr0 + (204 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp409 = tl.load(in_ptr0 + (205 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (206 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp413 = tl.load(in_ptr0 + (207 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp415 = tl.load(in_ptr0 + (208 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (209 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp419 = tl.load(in_ptr0 + (210 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp421 = tl.load(in_ptr0 + (211 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (212 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp425 = tl.load(in_ptr0 + (213 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp427 = tl.load(in_ptr0 + (214 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp429 = tl.load(in_ptr0 + (215 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp431 = tl.load(in_ptr0 + (216 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp433 = tl.load(in_ptr0 + (217 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp435 = tl.load(in_ptr0 + (218 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp437 = tl.load(in_ptr0 + (219 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp439 = tl.load(in_ptr0 + (220 + 256 * x0), xmask, eviction_policy=
        '