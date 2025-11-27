import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (42 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (43 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (44 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (45 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (46 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (47 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (48 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (49 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (50 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (51 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (52 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (53 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (54 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (55 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (56 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (57 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (58 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (59 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (60 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (61 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (62 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (63 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (64 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (65 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (66 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (67 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (68 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (69 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (70 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (71 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (72 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (73 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (74 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (75 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (76 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (77 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (78 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (79 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (80 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (81 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (82 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (83 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (84 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (85 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (86 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (87 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (88 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (89 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (90 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (91 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (92 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (93 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (94 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (95 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (96 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (97 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (98 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (99 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (100 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (101 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (102 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (103 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (104 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (105 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (106 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (107 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (108 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (109 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (110 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (111 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (112 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (113 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (114 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (115 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (116 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (117 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (118 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (119 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (120 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (121 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (122 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (123 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (124 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (125 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (126 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (127 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (128 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (129 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (130 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (131 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (132 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (133 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (134 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (135 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (136 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (137 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (138 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_ptr0 + (139 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (140 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp281 = tl.load(in_ptr0 + (141 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp283 = tl.load(in_ptr0 + (142 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (143 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (144 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (145 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (146 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp293 = tl.load(in_ptr0 + (147 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + (148 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (149 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (150 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp301 = tl.load(in_ptr0 + (151 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (152 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + (153 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp307 = tl.load(in_ptr0 + (154 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (155 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (156 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp313 = tl.load(in_ptr0 + (157 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (158 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (159 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp319 = tl.load(in_ptr0 + (160 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (161 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (162 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp325 = tl.load(in_ptr0 + (163 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (164 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp329 = tl.load(in_ptr0 + (165 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (166 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (167 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (168 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp337 = tl.load(in_ptr0 + (169 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (170 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp341 = tl.load(in_ptr0 + (171 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp343 = tl.load(in_ptr0 + (172 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (173 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (174 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp349 = tl.load(in_ptr0 + (175 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (176 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp353 = tl.load(in_ptr0 + (177 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp355 = tl.load(in_ptr0 + (178 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (179 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (180 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp361 = tl.load(in_ptr0 + (181 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (182 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp365 = tl.load(in_ptr0 + (183 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp367 = tl.load(in_ptr0 + (184 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (185 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp371 = tl.load(in_ptr0 + (186 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp373 = tl.load(in_ptr0 + (187 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (188 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp377 = tl.load(in_ptr0 + (189 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp379 = tl.load(in_ptr0 + (190 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (191 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp383 = tl.load(in_ptr0 + (192 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp385 = tl.load(in_ptr0 + (193 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (194 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp389 = tl.load(in_ptr0 + (195 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp391 = tl.load(in_ptr0 + (196 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (197 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp395 = tl.load(in_ptr0 + (198 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp397 = tl.load(in_ptr0 + (199 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (200 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp401 = tl.load(in_ptr0 + (201 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp403 = tl.load(in_ptr0 + (202 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (203 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp407 = tl.load(in_ptr0 + (204 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp409 = tl.load(in_ptr0 + (205 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (206 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp413 = tl.load(in_ptr0 + (207 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp415 = tl.load(in_ptr0 + (208 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (209 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp419 = tl.load(in_ptr0 + (210 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp421 = tl.load(in_ptr0 + (211 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (212 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp425 = tl.load(in_ptr0 + (213 + 8192 * x0), xmask, eviction