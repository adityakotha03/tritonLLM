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
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_mean_pow_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (8192 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (12288 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16384 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (20480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (24576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (28672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (32768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (36864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (40960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (45056 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (49152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (53248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (57344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (61440 + x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (65536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (69632 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (73728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (77824 + x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (81920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (86016 + x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (90112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (94208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (98304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (102400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (106496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (110592 + x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (114688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (118784 + x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (122880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (126976 + x0), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (131072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (135168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (139264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (143360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (147456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (151552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (155648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (159744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (163840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (167936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (172032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (176128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (180224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (184320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (188416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (192512 + x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (196608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (200704 + x0), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (204800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (208896 + x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (212992 + x0), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (217088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (221184 + x0), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (225280 + x0), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (229376 + x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (233472 + x0), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (237568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (241664 + x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (245760 + x0), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (249856 + x0), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr0 + (253952 + x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (258048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (262144 + x0), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr0 + (266240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (270336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (274432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr0 + (278528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (282624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (286720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr0 + (290816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (294912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (299008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr0 + (303104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (307200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (311296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr0 + (315392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (319488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (323584 + x0), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr0 + (327680 + x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (331776 + x0), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (335872 + x0), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr0 + (339968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (344064 + x0), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (348160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr0 + (352256 + x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (356352 + x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (360448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr0 + (364544 + x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (368640 + x0), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (372736 + x0), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr0 + (376832 + x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (380928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (385024 + x0), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr0 + (389120 + x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (393216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (397312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr0 + (401408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (405504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (409600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr0 + (413696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (417792 + x0), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (421888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr0 + (425984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (430080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (434176 + x0), xmask, eviction_policy='evict_last'
        )
    tmp215 = tl.load(in_ptr0 + (438272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp217 = tl.load(in_ptr0 + (442368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (446464 + x0), xmask, eviction_policy='evict_last'
        )
    tmp221 = tl.load(in_ptr0 + (450560 + x0), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr0 + (454656 + x0), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (458752 + x0), xmask, eviction_policy='evict_last'
        )
    tmp227 = tl.load(in_ptr0 + (462848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp229 = tl.load(in_ptr0 + (466944 + x0), xmask, eviction_policy='evict_last'
        )
    tmp231 = tl.load(in_ptr0 + (471040 + x0), xmask, eviction_policy='evict_last'
        )
    tmp233 = tl.load(in_ptr0 + (475136 + x0), xmask, eviction_policy='evict_last'
        )
    tmp235 = tl.load(in_ptr0 + (479232 + x0), xmask, eviction_policy='evict_last'
        )
    tmp237 = tl.load(in_ptr0 + (483328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp239 = tl.load(in_ptr0 + (487424 + x0), xmask, eviction_policy='evict_last'
        )
    tmp241 = tl.load(in_ptr0 + (491520 + x0), xmask, eviction_policy='evict_last'
        )
    tmp243 = tl.load(in_ptr0 + (495616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp245 = tl.load(in_ptr0 + (499712 + x0), xmask, eviction_policy='evict_last'
        )
    tmp247 = tl.load(in_ptr0 + (503808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp249 = tl.load(in_ptr0 + (507904 + x0), xmask, eviction_policy='evict_last'
        )
    tmp251 = tl.load(in_ptr0 + (511999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp253 = tl.load(in_ptr0 + (516095 + x0), xmask, eviction_policy='evict_last'
        )
    tmp255 = tl.load(in_ptr0 + (520191 + x0), xmask, eviction_policy='evict_last'
        )
    tmp257 = tl.load(in_ptr0 + (524287 + x0), xmask, eviction_policy='evict_last'
        )
    tmp259 = tl.load(in_ptr0 + (528383 + x0), xmask, eviction_policy='evict_last'
        )
    tmp261 = tl.load(in_ptr0 + (532479 + x0), xmask, eviction_policy='evict_last'
        )
    tmp263 = tl.load(in_ptr0 + (536575 + x0), xmask, eviction_policy='evict_last'
        )
    tmp265 = tl.load(in_ptr0 + (540671 + x0), xmask, eviction_policy='evict_last'
        )
    tmp267 = tl.load(in_ptr0 + (544767 + x0), xmask, eviction_policy='evict_last'
        )
    tmp269 = tl.load(in_ptr0 + (548863 + x0), xmask, eviction_policy='evict_last'
        )
    tmp271 = tl.load(in_ptr0 + (552959 + x0), xmask, eviction_policy='evict_last'
        )
    tmp273 = tl.load(in_ptr0 + (557055 + x0), xmask, eviction_policy='evict_last'
        )
    tmp275 = tl.load(in_ptr0 + (561151 + x0), xmask, eviction_policy='evict_last'
        )
    tmp277 = tl.load(in_ptr0 + (565247 + x0), xmask, eviction_policy='evict_last'
        )
    tmp279 = tl.load(in_ptr0 + (569343 + x0), xmask, eviction_policy='evict_last'
        )
    tmp281 = tl.load(in_ptr0 + (573439 + x0), xmask, eviction_policy='evict_last'
        )
    tmp283 = tl.load(in_ptr0 + (577535 + x0), xmask, eviction_policy='evict_last'
        )
    tmp285 = tl.load(in_ptr0 + (581631 + x0), xmask, eviction_policy='evict_last'
        )
    tmp287 = tl.load(in_ptr0 + (585727 + x0), xmask, eviction_policy='evict_last'
        )
    tmp289 = tl.load(in_ptr0 + (589823 + x0), xmask, eviction_policy='evict_last'
        )
    tmp291 = tl.load(in_ptr0 + (593919 + x0), xmask, eviction_policy='evict_last'
        )
    tmp293 = tl.load(in_ptr0 + (598015 + x0), xmask, eviction_policy='evict_last'
        )
    tmp295 = tl.load(in_ptr0 + (602111 + x0), xmask, eviction_policy='evict_last'
        )
    tmp297 = tl.load(in_ptr0 + (606207 + x0), xmask, eviction_policy='evict_last'
        )
    tmp299 = tl.load(in_ptr0 + (610303 + x0), xmask, eviction_policy='evict_last'
        )
    tmp301 = tl.load(in_ptr0 + (614399 + x0), xmask, eviction_policy='evict_last'
        )
    tmp303 = tl.load(in_ptr0 + (618495 + x0), xmask, eviction_policy='evict_last'
        )
    tmp305 = tl.load(in_ptr0 + (622591 + x0), xmask, eviction_policy='evict_last'
        )
    tmp307 = tl.load(in_ptr0 + (626687 + x0), xmask, eviction_policy='evict_last'
        )
    tmp309 = tl.load(in_ptr0 + (630783 + x0), xmask, eviction_policy='evict_last'
        )
    tmp311 = tl.load(in_ptr0 + (634879 + x0), xmask, eviction_policy='evict_last'
        )
    tmp313 = tl.load(in_ptr0 + (638975 + x0), xmask, eviction_policy='evict_last'
        )
    tmp315 = tl.load(in_ptr0 + (643071 + x0), xmask, eviction_policy='evict_last'
        )
    tmp317 = tl.load(in_ptr0 + (647167 + x0), xmask, eviction_policy='evict_last'
        )
    tmp319 = tl.load(in_ptr0 + (651263 + x0), xmask, eviction_policy='evict_last'
        )
    tmp321 = tl.load(in_ptr0 + (655359 + x0), xmask, eviction_policy='evict_last'
        )
    tmp323 = tl.load(in_ptr0 + (659455 + x0), xmask, eviction_policy='evict_last'
        )
    tmp325 = tl.load(in_ptr0 + (663551 + x0), xmask, eviction_policy='evict_last'
        )
    tmp327 = tl.load(in_ptr0 + (667647 + x0), xmask, eviction_policy='evict_last'
        )
    tmp329 = tl.load(in_ptr0 + (671743 + x0), xmask, eviction_policy='evict_last'
        )
    tmp331 = tl.load(in_ptr0 + (675839 + x0), xmask, eviction_policy='evict_last'
        )
    tmp333 = tl.load(in_ptr0 + (679935 + x0), xmask, eviction_policy='evict_last'
        )
    tmp335 = tl.load(in_ptr0 + (684031 + x0), xmask, eviction_policy='evict_last'
        )
    tmp337 = tl.load(in_ptr0 + (688127 + x0), xmask, eviction_policy='evict_last'
        )
    tmp339 = tl.load(in_ptr0 + (692223 + x0), xmask, eviction_policy='evict_last'
        )
    tmp341 = tl.load(in_ptr0 + (696319 + x0), xmask, eviction_policy='evict_last'
        )
    tmp343 = tl.load(in_ptr0 + (700415 + x0), xmask, eviction_policy='evict_last'
        )
    tmp345 = tl.load(in_ptr0 + (704511 + x0), xmask, eviction_policy='evict_last'
        )
    tmp347 = tl.load(in_ptr0 + (708607 + x0), xmask, eviction_policy='evict_last'
        )
    tmp349 = tl.load(in_ptr0 + (712703 + x0), xmask, eviction_policy='evict_last'
        )
    tmp351 = tl.load(in_ptr0 + (716799 + x0), xmask, eviction_policy='evict_last'
        )
    tmp353 = tl.load(in_ptr0 + (720895 + x0), xmask, eviction_policy='evict_last'
        )
    tmp355 = tl.load(in_ptr0 + (724991 + x0), xmask, eviction_policy='evict_last'
        )
    tmp357 = tl.load(in_ptr0 + (729087 + x0), xmask, eviction_policy='evict_last'
        )
    tmp359 = tl.load(in_ptr0 + (733183 + x0), xmask, eviction_policy='evict_last'
        )
    tmp361 = tl.load(in_ptr0 + (737279 + x0), xmask, eviction_policy='evict_last'
        )
    tmp363 = tl.load(in_ptr0 + (741375 + x0), xmask, eviction_policy='evict_last'
        )
    tmp365 = tl.load(in_ptr0 + (745471 + x0), xmask, eviction_policy='evict_last'
        )
    tmp367 = tl.load(in_ptr0 + (749567 + x0), xmask, eviction_policy='evict_last'
        )
    tmp369 = tl.load(in_ptr0 + (753663 + x0), xmask, eviction_policy='evict_last'
        )
    tmp371 = tl.load(in_ptr0 + (757759 + x0), xmask, eviction_policy='evict_last'
        )
    tmp373 = tl.load(in_ptr0 + (761855 + x0), xmask, eviction_policy='evict_last'
        )
    tmp375 = tl.load(in_ptr0 + (765951 + x0), xmask, eviction_policy='evict_last'
        )
    tmp377 = tl.load(in_ptr0 + (769947 + x0), xmask, eviction_policy='evict_last'
        )
    tmp379 = tl.load(in_ptr0 + (774043 + x0), xmask, eviction_policy='evict_last'
        )
    tmp381 = tl.load(in_ptr0 + (778139 + x0), xmask, eviction_policy='evict_last'
        )
    tmp383 = tl.load(in_ptr0 + (782235 + x0), xmask, eviction_policy='evict_last'
        )
    tmp385 = tl.load(in_ptr0 + (786331 + x0), xmask, eviction_policy='evict_last'
        )
    tmp387 = tl.load(in_ptr0 + (790427 + x0), xmask, eviction_policy='evict_last'
        )
    tmp389 = tl.load(in_ptr0 + (794523 + x0), xmask, eviction_policy='evict_last'
        )
    tmp391 = tl.load(in_ptr0 + (798619 + x0), xmask, eviction_policy='evict_last'
        )
    tmp393 = tl.load(in_ptr0 + (802715 + x0), xmask, eviction_policy='evict_last'
        )
    tmp395 = tl.load(in_ptr0 + (806811 + x0), xmask, eviction_policy='evict_last'
        )
    tmp397 = tl.load(in_ptr0 + (810907 + x0), xmask, eviction_policy='evict_last'
        )
    tmp399 = tl.load(in_ptr0 + (814999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp401 = tl.load(in_ptr0 + (819095 + x0), xmask, eviction_policy='evict_last'
        )
    tmp403 = tl.load(in_ptr0 + (823191 + x0), xmask, eviction_policy='evict_last'
        )
    tmp405 = tl.load(in_ptr0 + (827287 + x0), xmask, eviction_policy='evict_last'
        )
    tmp407 = tl.load(in_ptr0 + (831383 + x0), xmask, eviction_policy='evict_last'
        )
    tmp409 = tl.load(in_ptr0 + (835479 + x0), xmask, eviction_policy='evict_last'
        )
    tmp411 = tl.load(in_ptr0 + (839575 + x0), xmask, eviction_policy='evict_last'
        )
    tmp413 = tl.load(in_ptr0 + (843671 + x0), xmask, eviction_policy='evict_last'
        )
    tmp415 = tl.load(in_ptr0 + (847767 + x0), xmask, eviction_policy='evict_last'
        )
    tmp417 = tl.load(in_ptr0 + (851863 + x0), xmask, eviction_policy='evict_last'
        )
    tmp419 = tl.load(in_ptr0 + (855959 + x0), xmask, eviction_policy='evict_last'
        )
    tmp421 = tl.load(in_ptr0 + (860055 + x0), xmask, eviction_policy='evict_last'
        )
    tmp423 = tl.load(in_ptr0 + (864151 + x0), xmask, eviction_policy='evict_last'
        )
    tmp425 = tl.load(in_ptr0 + (868247 + x0), xmask, eviction_policy='evict_last'
        )
    tmp427 = tl.load(in_ptr0 + (872343 + x0), xmask, eviction_policy='evict_last'
        )
    tmp429 = tl.load(in_ptr0 + (876439 + x0), xmask, eviction_policy='evict_last'
        )
    tmp431 = tl.load(in_ptr0 + (880535 + x0), xmask, eviction_policy='evict_last'
        )
    tmp433 = tl.load(in_ptr0 + (884631 + x0), xmask, eviction_policy='evict_last'
        )
    tmp435 = tl.load(in_ptr0 + (888727 + x0), xmask, eviction_policy='evict_last'
        )
    tmp437 = tl.load(in_ptr0 + (892823 + x0), xmask, eviction_policy='evict_last'
        )
    tmp439 = tl.load(in_ptr0