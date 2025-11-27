import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool1d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = xindex // 32768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32768 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 32768 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32768 + x0 + 32768 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (32769 + x0 + 32768 * x1), xmask)
    tmp2 = tmp1 > tmp0
    tmp4 = tmp3 > tmp1
    tmp6 = tmp5 > tmp3
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tl.where(tmp4, tmp7, tmp9)
    tmp11 = tl.where(tmp6, tmp7, tmp10)
    tl.store(out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 32768 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (42 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (43 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (44 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (45 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (46 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (47 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (48 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (49 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (50 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (51 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (52 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (53 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (54 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (55 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (56 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (57 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (58 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (59 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (60 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (61 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (62 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (63 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (64 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (65 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (66 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (67 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (68 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (69 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (70 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (71 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (72 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (73 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (74 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (75 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (76 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (77 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (78 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (79 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (80 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (81 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (82 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (83 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (84 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (85 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (86 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (87 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (88 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (89 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (90 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (91 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (92 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (93 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (94 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (95 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (96 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (97 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (98 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (99 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (100 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (101 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (102 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (103 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (104 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (105 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (106 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (107 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (108 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (109 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (110 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (111 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (112 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (113 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (114 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (115 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (116 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (117 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (118 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (119 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (120 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (121 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (122 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (123 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (124 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (125 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (126 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (127 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (128 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (129 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (130 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (131 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (132 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (133 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (134 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (135 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (136 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (137 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (138 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_ptr0 + (139 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (140 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp281 = tl.load(in_ptr0 + (141 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp283 = tl.load(in_ptr0 + (142 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (143 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (144 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (145 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (146 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp293 = tl.load(in_ptr0 + (147 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + (148 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (149 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (150 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp301 = tl.load(in_ptr0 + (151 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (152 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + (153 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp307 = tl.load(in_ptr0 + (154 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (155 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (156 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp313 = tl.load(in_ptr0 + (157 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (158 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (159 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp319 = tl.load(in_ptr0 + (160 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (161 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (162 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp325 = tl.load(in_ptr0 + (163 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (164 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp329 = tl.load(in_ptr0 + (165 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (166 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (167 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (168 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp337 = tl.load(in_ptr0 + (169 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (170 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp341 = tl.load(in_ptr0 + (171 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp343 = tl.load(in_ptr0 + (172 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (173 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (174 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp349 = tl.load(in_ptr0 + (175 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (176 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp353 = tl.load(in_ptr0 + (177 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp355 = tl.load(in_ptr0 + (178 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (179 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (180 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp361 = tl.load(in_ptr0 + (181 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (182 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp365 = tl.load(in_ptr0 + (183 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp367 = tl.load(in_ptr0 + (184 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (185 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp371 = tl.load(in_ptr0 + (186 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp373 = tl.load(in_ptr0 + (187 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (188 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp377 = tl.load(in_ptr0 + (189 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp379 = tl.load(in_ptr0 + (190 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (191 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp383 = tl.load(in_ptr0 + (192 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp385 = tl.load(in_ptr0 + (193 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (194 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp389 = tl.load(in_ptr0 + (195 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp391 = tl.load(in_ptr0 + (196 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (197 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp395 = tl.load(in_ptr0 + (198 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp397 = tl.load(in_ptr0 + (199 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp399