import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_hardswish_relu_threshold_backward_0(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_out_ptr0 + (1048576 + x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + 1)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_out_ptr0 + (2097152 + x0), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + 2)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_out_ptr0 + (3145728 + x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + 3)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp19 = tl.load(in_out_ptr0 + (4194304 + x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + 4)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp24 = tl.load(in_out_ptr0 + (5242880 + x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + 5)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp29 = tl.load(in_out_ptr0 + (6291456 + x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + 6)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp34 = tl.load(in_out_ptr0 + (7340032 + x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + 7)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp39 = tl.load(in_out_ptr0 + (8388608 + x0), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + 8)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp44 = tl.load(in_out_ptr0 + (9437184 + x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + 9)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp49 = tl.load(in_out_ptr0 + (10485760 + x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + 10)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp54 = tl.load(in_out_ptr0 + (11534368 + x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + 11)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp59 = tl.load(in_out_ptr0 + (12582944 + x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + 12)
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
    tmp64 = tl.load(in_out_ptr0 + (13631520 + x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + 13)
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp69 = tl.load(in_out_ptr0 + (14680096 + x0), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + 14)
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
    tmp74 = tl.load(in_out_ptr0 + (15728672 + x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + 15)
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp79 = tl.load(in_out_ptr0 + (16777248 + x0), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + 16)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
    tmp84 = tl.load(in_out_ptr0 + (17825824 + x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + 17)
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK])
    tmp89 = tl.load(in_out_ptr0 + (18874392 + x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + 18)
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp94 = tl.load(in_out_ptr0 + (19922968 + x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + 19)
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp99 = tl.load(in_out_ptr0 + (20971520 + x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + 20)
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK])
    tmp104 = tl.load(in_out_ptr0 + (22020096 + x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + 21)
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK])
    tmp109 = tl.load(in_out_ptr0 + (23068672 + x0), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + 22)
    tmp111 = tl.broadcast_to(tmp110, [XBLOCK])
    tmp114 = tl.load(in_out_ptr0 + (24117248 + x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + 23)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK])
    tmp119 = tl.load(in_out_ptr0 + (25165824 + x0), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + 24)
    tmp121 = tl.broadcast_to(tmp120, [XBLOCK])
    tmp124 = tl.load(in_out_ptr0 + (26214400 + x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + 25)
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK])
    tmp129 = tl.load(in_out_ptr0 + (27262976 + x0), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + 26)
    tmp131 = tl.broadcast_to(tmp130, [XBLOCK])
    tmp134 = tl.load(in_out_ptr0 + (28311552 + x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + 27)
    tmp136 = tl.broadcast_to(tmp135, [XBLOCK])
    tmp139 = tl.load(in_out_ptr0 + (29360128 + x0), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + 28)
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK])
    tmp144 = tl.load(in_out_ptr0 + (30408704 + x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + 29)
    tmp146 = tl.broadcast_to(tmp145, [XBLOCK])
    tmp149 = tl.load(in_out_ptr0 + (31457280 + x0), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + 30)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK])
    tmp154 = tl.load(in_out_ptr0 + (32505856 + x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + 31)
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK])
    tmp159 = tl.load(in_out_ptr0 + (33554432 + x0), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + 32)
    tmp161 = tl.broadcast_to(tmp160, [XBLOCK])
    tmp164 = tl.load(in_out_ptr0 + (34602992 + x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + 33)
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK])
    tmp169 = tl.load(in_out_ptr0 + (35651568 + x0), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + 34)
    tmp171 = tl.broadcast_to(tmp170, [XBLOCK])
    tmp174 = tl.load(in_out_ptr0 + (36700144 + x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + 35)
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK])
    tmp179 = tl.load(in_out_ptr0 + (37748720 + x0), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + 36)
    tmp181 = tl.broadcast_to(tmp180, [XBLOCK])
    tmp184 = tl.load(in_out_ptr0 + (38797296 + x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + 37)
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK])
    tmp189 = tl.load(in_out_ptr0 + (39845872 + x0), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + 38)
    tmp191 = tl.broadcast_to(tmp190, [XBLOCK])
    tmp194 = tl.load(in_out_ptr0 + (40894448 + x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + 39)
    tmp196 = tl.broadcast_to(tmp195, [XBLOCK])
    tmp199 = tl.load(in_out_ptr0 + (41943024 + x0), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + 40)
    tmp201 = tl.broadcast_to(tmp200, [XBLOCK])
    tmp204 = tl.load(in_out_ptr0 + (42991600 + x0), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + 41)
    tmp206 = tl.broadcast_to(tmp205, [XBLOCK])
    tmp209 = tl.load(in_out_ptr0 + (44040176 + x0), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + 42)
    tmp211 = tl.broadcast_to(tmp210, [XBLOCK])
    tmp214 = tl.load(in_out_ptr0 + (45088752 + x0), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + 43)
    tmp216 = tl.broadcast_to(tmp215, [XBLOCK])
    tmp219 = tl.load(in_out_ptr0 + (46137328 + x0), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + 44)
    tmp221 = tl.broadcast_to(tmp220, [XBLOCK])
    tmp224 = tl.load(in_out_ptr0 + (47185904 + x0), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + 45)
    tmp226 = tl.broadcast_to(tmp225, [XBLOCK])
    tmp229 = tl.load(in_out_ptr0 + (48234480 + x0), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + 46)
    tmp231 = tl.broadcast_to(tmp230, [XBLOCK])
    tmp234 = tl.load(in_out_ptr0 + (49283056 + x0), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + 47)
    tmp236 = tl.broadcast_to(tmp235, [XBLOCK])
    tmp239 = tl.load(in_out_ptr0 + (50331632 + x0), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + 48)
    tmp241 = tl.broadcast_to(tmp240, [XBLOCK])
    tmp244 = tl.load(in_out_ptr0 + (51380208 + x0), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + 49)
    tmp246 = tl.broadcast_to(tmp245, [XBLOCK])
    tmp249 = tl.load(in_out_ptr0 + (52428800 + x0), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + 50)
    tmp251 = tl.broadcast_to(tmp250, [XBLOCK])
    tmp254 = tl.load(in_out_ptr0 + (53477376 + x0), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + 51)
    tmp256 = tl.broadcast_to(tmp255, [XBLOCK])
    tmp259 = tl.load(in_out_ptr0 + (54525952 + x0), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + 52)
    tmp261 = tl.broadcast_to(tmp260, [XBLOCK])
    tmp264 = tl.load(in_out_ptr0 + (55574528 + x0), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + 53)
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK])
    tmp269 = tl.load(in_out_ptr0 + (56623104 + x0), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + 54)
    tmp271 = tl.broadcast_to(tmp270, [XBLOCK])
    tmp274 = tl.load(in_out_ptr0 + (57671680 + x0), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + 55)
    tmp276 = tl.broadcast_to(tmp275, [XBLOCK])
    tmp279 = tl.load(in_out_ptr0 + (58720256 + x0), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + 56)
    tmp281 = tl.broadcast_to(tmp280, [XBLOCK])
    tmp284 = tl.load(in_out_ptr0 + (59768832 + x0), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + 57)
    tmp286 = tl.broadcast_to(tmp285, [XBLOCK])
    tmp289 = tl.load(in_out_ptr0 + (60817408 + x0), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + 58)
    tmp291 = tl.broadcast_to(tmp290, [XBLOCK])
    tmp294 = tl.load(in_out_ptr0 + (61865984 + x0), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + 59)
    tmp296 = tl.broadcast_to(tmp295, [XBLOCK])
    tmp299 = tl.load(in_out_ptr0 + (62914560 + x0), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + 60)
    tmp301 = tl.broadcast_to(tmp300, [XBLOCK])
    tmp304 = tl.load(in_out_ptr0 + (63963136 + x0), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + 61)
    tmp306 = tl.broadcast_to(tmp305, [XBLOCK])
    tmp309 = tl.load(in_out_ptr0 + (65011712 + x0), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + 62)
    tmp311 = tl.broadcast_to(tmp310, [XBLOCK])
    tmp314 = tl.load(in_out_ptr0 + (66060288 + x0), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + 63)
    tmp316 = tl.broadcast_to(tmp315, [XBLOCK])
    tmp17 = tmp0 * tmp2
    tmp18 = tmp17 > 0
    tmp19_1 = triton_helpers.maximum(tmp17, tmp18)
    tmp20_1 = triton_helpers.minimum(tmp19_1, 3.0)
    tmp21_1 = tmp19_1 * tmp6
    tmp22 = tmp21_1 > 0
    tmp23 = tmp21_1 * tmp11
    tmp24_1 = tmp23 > 0
    tmp25_1 = triton_helpers.maximum(tmp23, tmp24_1)
    tmp26_1 = triton_helpers.minimum(tmp25_1, 3.0)
    tmp27 = tmp25_1 * tmp16
    tmp28 = tmp27 > 0
    tmp29_1 = tmp27 * tmp21
    tmp30_1 = tmp29_1 > 0
    tmp31_1 = triton_helpers.maximum(tmp29_1, tmp30_1)
    tmp32 = tmp31_1 * tmp26
    tmp33 = tmp32 > 0
    tmp34_1 = tmp32 * tmp31
    tmp35_1 = tmp34_1 > 0
    tmp36_1 = triton_helpers.maximum(tmp34_1, tmp35_1)
    tmp37 = tmp36_1 * tmp36
    tmp38 = tmp37 > 0
    tmp39_1 = tmp37 * tmp41
    tmp40_1 = tmp39_1 > 0
    tmp41_1 = triton_helpers.maximum(tmp39_1, tmp40_1)
    tmp42 = tmp41_1 * tmp46
    tmp43 = tmp42 > 0
    tmp44_1 = tmp42 * tmp51
    tmp45_1 = tmp44_1 > 0
    tmp46_1 = triton_helpers.maximum(tmp44_1, tmp45_1)
    tmp47 = tmp46_1 * tmp56
    tmp48 = tmp47 > 0
    tmp49_1 = tmp47 * tmp61
    tmp50_1 = tmp49_1 > 0
    tmp51_1 = triton_helpers.maximum(tmp49_1, tmp50_1)
    tmp52 = tmp51_1 * tmp66
    tmp53 = tmp52 > 0
    tmp54_1 = tmp52 * tmp71
    tmp55_1 = tmp54_1 > 0
    tmp56_1 = triton_helpers.maximum(tmp54_1, tmp55_1)
    tmp57 = tmp56_1 * tmp76
    tmp58 = tmp57 > 0
    tmp59_1 = tmp57 * tmp81
    tmp60_1 = tmp59_1 > 0
    tmp61_1 = triton_helpers.maximum(tmp59_1, tmp60_1)
    tmp62 = tmp61_1 * tmp86
    tmp63 = tmp62 > 0
    tmp64_1 = tmp62 * tmp91
    tmp65_1 = tmp64_1 > 0
    tmp66_1 = triton_helpers.maximum(tmp64_1, tmp65_1)
    tmp67 = tmp66_1 * tmp96
    tmp68 = tmp67 > 0
    tmp69_1 = tmp67 * tmp101
    tmp70_1 = tmp69_1 > 0
    tmp71_1 = triton_helpers.maximum(tmp69_1, tmp70_1)
    tmp72 = tmp71_1 * tmp106
    tmp73 = tmp72 > 0
    tmp74_1 = tmp72 * tmp111
    tmp75_1 = tmp74_1 > 0
    tmp76_1 = triton_helpers.maximum(tmp74_1, tmp75_1)
    tmp77 = tmp76_1 * tmp116
    tmp78 = tmp77 > 0
    tmp79_1 = tmp77 * tmp121
    tmp80_1 = tmp79_1 > 0
    tmp81_1 = triton_helpers.maximum(tmp79_1, tmp80_1)
    tmp82 = tmp81_1 * tmp126
    tmp83 = tmp82 > 0
    tmp84_1 = tmp82 * tmp131
    tmp85_1 = tmp84_1 > 0
    tmp86_1 = triton_helpers.maximum(tmp84_1, tmp85_1)
    tmp87 = tmp86_1 * tmp136
    tmp88 = tmp87 > 0
    tmp89_1 = tmp87 * tmp141
    tmp90_1 = tmp89_1 > 0
    tmp91_1 = triton_helpers.maximum(tmp89_1, tmp90_1)
    tmp92 = tmp91_1 * tmp146
    tmp93 = tmp92 > 0
    tmp94_1 = tmp92 * tmp151
    tmp95_1 = tmp94_1 > 0
    tmp96_1 = triton_helpers.maximum(tmp94_1, tmp95_1)
    tmp97 = tmp96_1 * tmp156
    tmp98 = tmp97 > 0
    tmp99_1 = tmp97 * tmp161
    tmp100_1 = tmp99_1 > 0
    tmp101_1 = triton_helpers.maximum(tmp99_1, tmp100_1)
    tmp102 = tmp101_1 * tmp166
    tmp103 = tmp102 > 0
    tmp104_1 = tmp102 * tmp171
    tmp105_1 = tmp104_1 > 0
    tmp106_1 = triton_helpers.maximum(tmp104_1, tmp105_1)
    tmp107 = tmp106_1 * tmp176
    tmp108 = tmp107 > 0
    tmp109_1 = tmp107 * tmp181
    tmp110_1 = tmp109_1 > 0
    tmp111_1 = triton_helpers.maximum(tmp109_1, tmp110_1)
    tmp112 = tmp111_1 * tmp186
    tmp113 = tmp112 > 0
    tmp114_1 = tmp112 * tmp191
    tmp115_1 = tmp114_1 > 0
    tmp116_1 = triton_helpers.maximum(tmp114_1, tmp115_1)
    tmp117 = tmp116_1 * tmp196
    tmp118 = tmp117 > 0
    tmp119_1 = tmp117 * tmp201
    tmp120_1 = tmp119_1 > 0
    tmp121_1 = triton_helpers.maximum(tmp119_1, tmp120_1)
    tmp122 = tmp121_1 * tmp206
    tmp123 = tmp122 > 0
    tmp124_1 = tmp122 * tmp211
    tmp125_1 = tmp124_1 > 0
    tmp126_1 = triton_helpers.maximum(tmp124_1, tmp125_1)
    tmp127 = tmp126_1 * tmp216
    tmp128 = tmp127 > 0
    tmp129_1 = tmp127 * tmp221
    tmp130_1 = tmp129_1 > 0
    tmp131_1 = triton_helpers.maximum(tmp129_1, tmp130_1)
    tmp132 = tmp131_1 * tmp226
    tmp133 = tmp132 > 0
    tmp134_1 = tmp132 * tmp231
    tmp135_1 = tmp134_1 > 0
    tmp136_1 = triton_helpers.maximum(tmp134_1, tmp135_1)
    tmp137 = tmp136_1 * tmp236
    tmp138 = tmp137 > 0
    tmp139_1 = tmp137 * tmp241
    tmp140_1 = tmp139_1 > 0
    tmp141_1 = triton_helpers.maximum(tmp139_1, tmp140_1)
    tmp142 = tmp141_1 * tmp246
    tmp143 = tmp142 > 0
    tmp144_1 = tmp142 * tmp251
    tmp145_1 = tmp144_1 > 0
    tmp146_1 = triton_helpers.maximum(tmp144_1, tmp145_1)
    tmp147 = tmp146_1 * tmp256
    tmp148 = tmp147 > 0
    tmp149_1 = tmp147 * tmp261
    tmp150_1 = tmp149_1 > 0
    tmp151_1 = triton_helpers.maximum(tmp149_1, tmp150_1)
    tmp152 = tmp151_1 * tmp266
    tmp153 = tmp152 > 0
    tmp154_1 = tmp152 * tmp271
    tmp155_1 = tmp154_1 > 0
    tmp156_1 = triton_helpers.maximum(tmp154_1, tmp155_1)
    tmp157 = tmp156_1 * tmp276
    tmp158 = tmp157 > 0
    tmp159_1 = tmp157 * tmp281
    tmp160_1 = tmp159_1 > 0
    tmp161_1 = triton_helpers.maximum(tmp159_1, tmp160_1)
    tmp162 = tmp161_1 * tmp286
    tmp163 = tmp162 > 0
    tmp164_1 = tmp162 * tmp291
    tmp165_1 = tmp164_1 > 0
    tmp166_1 = triton_helpers.maximum(tmp164_1, tmp165_1)
    tmp167 = tmp166_1 * tmp296
    tmp168 = tmp167 > 0
    tmp169_1 = tmp167 * tmp301
    tmp170_1 = tmp169_1 > 0
    tmp171_1 = triton_helpers.maximum(tmp169_1, tmp170_1)
    tmp172 = tmp171_1 * tmp306
    tmp173 = tmp172 > 0
    tmp174_1 = tmp172 * tmp311
    tmp175_1 = tmp174_1 > 0
    tmp176_1 = triton_helpers.maximum(tmp174_1, tmp175_1)
    tmp177 = tmp176_1 * tmp316
    tmp178 = tmp177 > 0
    tmp179_1 = tmp177 * tmp321
    tmp180_1 = tmp179_1 > 0
    tmp181_1 = triton_helpers.maximum(tmp179_1, tmp180_1)
    tmp182 = tmp181_1 * tmp326
    tmp183 = tmp182 > 0
    tmp184_1 = tmp182 * tmp331
    tmp185_1 = tmp184_1 > 0
    tmp186_1 = triton_helpers.maximum(tmp184_1, tmp185_1)
    tmp187 = tmp186_1 * tmp336
    tmp188 = tmp187 > 0
    tmp189_1 = tmp187 * tmp341
    tmp190_1 = tmp189_1 > 0
    tmp191_1 = triton_helpers.maximum(tmp189_1, tmp190_1)
    tmp192 = tmp191_1 * tmp346
    tmp193 = tmp192 > 0
    tmp194_1 = tmp192 * tmp351
    tmp195_1 = tmp194_1 > 0
    tmp196_1 = triton_helpers.maximum(tmp194_1, tmp195_1)
    tmp197 = tmp196_1 * tmp356
    tmp198 = tmp197 > 0
    tmp199_1 = tmp197 * tmp361
    tmp200_1 = tmp199_1 > 0
    tmp201_1 = triton_helpers.maximum(tmp199_1, tmp200_1)
    tmp202 = tmp201_1 * tmp366
    tmp203 = tmp202 > 0
    tmp204_1 = tmp202 * tmp371
    tmp205_1 = tmp204_1 > 0
    tmp206_1 = triton_helpers.maximum(tmp204_1, tmp205_1)
    tmp207 = tmp206_1 * tmp376
    tmp208 = tmp207 > 0
    tmp209_1 = tmp207 * tmp381
    tmp210_1 = tmp209_1 > 0
    tmp211_1 = triton_helpers.maximum(tmp209_1, tmp210_1)
    tmp212 = tmp211_1 *