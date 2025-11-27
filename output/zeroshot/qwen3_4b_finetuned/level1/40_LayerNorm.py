import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (64 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (65 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (66 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (67 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (68 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (69 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (128 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (129 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (130 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (131 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (132 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (133 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (192 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (193 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (194 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (195 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (196 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (197 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (256 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (257 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (258 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (259 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (260 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (261 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (320 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (321 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (322 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (323 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (324 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (325 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (384 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (385 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (386 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (387 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (388 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (389 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (448 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (449 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (450 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (451 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (452 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (453 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (512 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (513 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (514 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (515 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (516 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (517 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (576 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (577 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (578 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (579 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (580 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (581 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (640 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (641 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (642 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (643 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (644 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (645 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (704 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (705 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (706 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (707 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (708 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (709 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (768 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (769 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (770 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (771 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (772 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (773 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (832 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (833 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (834 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (835 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (836 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (837 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (896 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (897 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (898 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (899 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (900 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (901 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (960 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (961 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (962 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (963 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (964 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (965 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (1024 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (1025 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (1026 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (1027 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (1028 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (1029 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (1088 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (1089 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (1090 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (1091 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (1092 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (1093 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (1152 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (1153 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (1154 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (1155 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (1156 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (1157 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (1216 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (1217 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (1218 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (1219 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (1220 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (1221 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (1280 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (1281 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (1282 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (1283 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (1284 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (1285 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (1344 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (1345 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (1346 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (1347 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (1348 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (1349 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (1408 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (1409 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (1410 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (1411 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (1412 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_ptr0 + (1413 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (1472 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (1473 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_ptr0 + (1474 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + (1475 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (1476 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp284 = tl.load(in_ptr0 + (1477 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (1536 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (1537 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + (1538 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp292 = tl.load(in_ptr0 + (1539 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (1540 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (1541 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (1600 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (1601 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp302 = tl.load(in_ptr0 + (1602 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp304 = tl.load(in_ptr0 + (1603 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (1604 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp308 = tl.load(in_ptr0 + (1605 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (1664 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (1665 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp314 = tl.load(in_ptr0 + (1666 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp316 = tl.load(in_ptr0 + (1667 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (1668 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp320 = tl.load(in_ptr0 + (1669 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (1728 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (1729 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp326 = tl.load(in_ptr0 + (1730 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp328 = tl.load(in_ptr0 + (1731 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (1732 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp332 = tl.load(in_ptr0 + (1733 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (1792 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (1793 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp338 = tl.load(in_ptr0 + (1794 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp340 = tl.load(in_ptr0 + (1795 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (1796 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp344 = tl.load(in_ptr0 + (1797 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (1856 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (1857 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp350 = tl.load(in_ptr0 + (1858 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp352 = tl.load(in_ptr0 + (1859 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (1860 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp356 = tl.load(in_ptr0 + (1861 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (1920 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (1921 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp362 = tl.load(in_ptr0 + (1922 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp364 = tl.load(in_ptr0 + (1923 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (1924 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp368 = tl.load(in_ptr0 + (1925 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp371 = tl.load(in_ptr0 + (1984 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (1985 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp374 = tl.load(in_ptr0 + (1986 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp376 = tl.load(in_ptr0 + (1987 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (1988 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp380 = tl.load(in_ptr0 + (1989 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp383 = tl.load(in_ptr0 + (2048 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (2049 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp386 = tl.load(in_ptr0 + (2050 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp388 = tl.load(in_ptr0 + (2051 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (2052 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp392 = tl.load(in_ptr0 + (2053 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp395 = tl.load(in_ptr0 + (2112 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (2113 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp398 = tl.load(in_ptr0 + (2114 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp400 = tl.load(in_ptr0 + (2115 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (2116 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp404 = tl.load(in_ptr0 + (2117 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp407 = tl.load(in_ptr0 + (2176 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (2177 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp410 = tl.load(in_ptr0 + (2178 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp412 = tl.load(in_ptr0 + (2179 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp414 = tl.load(in_ptr0 + (2180 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp416 = tl.load(in_ptr0 + (2181 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp419 = tl.load(in_ptr0 + (2240 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp420 = tl.load(in_ptr0 + (2241 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp422 = tl.load(in_ptr0 + (2242 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp424 = tl.load(in_ptr0 + (2243 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp426 = tl.load(in_ptr0 + (2244 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp428 = tl.load(in_ptr0 + (2245 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp431 = tl.load(in_ptr0 + (2304 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp432 = tl.load(in_ptr0 + (2305 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp434 = tl.load(in_ptr0 + (2306 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp436 = tl.load(in_ptr0 + (2307 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp438 = tl.load(in_ptr0 + (2308 + 64 * x0),