import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool1d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_gelu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.0
    tmp8 = tmp6 > tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp6, tmp9)
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (8192 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (24576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (32768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (40960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (49152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (57344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr0 + (65536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (73728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr0 + (81920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (90112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr0 + (98304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (106496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (114688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (122880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (131072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (139264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (147456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (155648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (163840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (172032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr0 + (180224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (188416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (196608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (204800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp26 = tl.load(in_ptr0 + (213008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (221200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr0 + (229408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (237600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr0 + (245808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (254000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp32 = tl.load(in_ptr0 + (262208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (270400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr0 + (278608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (286800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr0 + (295008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (303200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr0 + (311408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (319600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr0 + (327808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (336000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (344208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (352400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp44 = tl.load(in_ptr0 + (360608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (368800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr0 + (377008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (385200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (393408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (401600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp50 = tl.load(in_ptr0 + (409808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (418000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr0 + (426208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (434400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (442608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (450800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr0 + (459008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (467200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr0 + (475408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (483600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (491808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (500000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp62 = tl.load(in_ptr0 + (508208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (516400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr0 + (524608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (532800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (541008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (549200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp68 = tl.load(in_ptr0 + (557408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (565600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr0 + (573808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (582000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (590208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (598400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp74 = tl.load(in_ptr0 + (606608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (614800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr0 + (623008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (631200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (639408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (647600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr0 + (655808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (664000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr0 + (672208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (680400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (688608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (696800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp86 = tl.load(in_ptr0 + (705008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (713200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr0 + (721408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (729600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (737808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (746000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp92 = tl.load(in_ptr0 + (754208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (762400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr0 + (770608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (778800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (787008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (795200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp98 = tl.load(in_ptr0 + (803408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (811600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr0 + (819808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (828000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (836208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (844400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr0 + (852608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (860800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr0 + (869008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (877200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (885408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (893600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp110 = tl.load(in_ptr0 + (901808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (910000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr0 + (918208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (926400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (934608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (942800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp116 = tl.load(in_ptr0 + (951008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (959200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr0 + (967408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (975600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (983808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (992000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp122 = tl.load(in_ptr0 + (1000208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (1008400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr0 + (1016608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr0 + (1024800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (1033008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (1041200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr0 + (1049408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (1057600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr0 + (1065808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr0 + (1074000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (1082208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (1090400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp134 = tl.load(in_ptr0 + (1098608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (1106800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr0 + (1115008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr0 + (1123200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (1131408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (1139600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp140 = tl.load(in_ptr0 + (1147808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (1156000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr0 + (1164208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr0 + (1172400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (1180608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (1188800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp146 = tl.load(in_ptr0 + (1197008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (1205200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr0 + (1213408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr0 + (1221600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (1229808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (1238000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr0 + (1246208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (1254400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr0 + (1262608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr0 + (1270800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (1279008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (1287200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp158 = tl.load(in_ptr0 + (1295408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (1303600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr0 + (1311808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr0 + (1320000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (1328208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (1336400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp164 = tl.load(in_ptr0 + (1344608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (1352800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr0 + (1361008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr0 + (1369200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (1377408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (1385600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp170 = tl.load(in_ptr0 + (1393808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (1402000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr0 + (1410208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr0 + (1418400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (1426608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (1434800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr0 + (1443008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (1451200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr0 + (1459408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr0 + (1467600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (1475808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (1484000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp182 = tl.load(in_ptr0 + (1492208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (1500400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr0 + (1508608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr0 + (1516800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (1525008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (1533200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp188 = tl.load(in_ptr0 + (1541408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (1549600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr0 + (1557808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr0 + (1566000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (1574208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (1582400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp194 = tl.load(in_ptr0 + (1590608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (1598800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr0 + (1607008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr0 + (1615200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (1623408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (1631600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr0 + (1639808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (1648000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr0 + (1656208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr0 + (1664400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (1672608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (1680800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp206 = tl.load(in_ptr0 + (1689008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (1697200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr0 + (1705408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr0 + (1713600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (1721808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (1730000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp212 = tl.load(in_ptr0 + (1738208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (1746400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl.load(in_ptr0 + (