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
def triton_poi_fused_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_mean_sqrt_2(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (4 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (5 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (6 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (7 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (8 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (9 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (10 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (11 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (12 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (13 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (14 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (15 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (16 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (17 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (18 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (19 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (20 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (21 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (22 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (23 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (24 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (25 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (26 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (27 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (28 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (29 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (30 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (31 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (32 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (33 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (34 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (35 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (36 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (37 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (38 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (39 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp3
    tmp4 = tmp2 + tmp6
    tmp5 = tmp4 + tmp9
    tmp6a = tmp5 + tmp12
    tmp7 = tmp6a + tmp15
    tmp8 = tmp7 + tmp18
    tmp9a = tmp8 + tmp21
    tmp10 = tmp9a + tmp24
    tmp11 = tmp10 + tmp27
    tmp12a = tmp11 + tmp30
    tmp13 = tmp12a + tmp33
    tmp14 = tmp13 + tmp36
    tmp15a = tmp14 + tmp39
    tmp16 = tmp15a + tmp42
    tmp17 = tmp16 + tmp45
    tmp18a = tmp17 + tmp48
    tmp19 = tmp18a + tmp51
    tmp20 = tmp19 + tmp54
    tmp21a = tmp20 + tmp57
    tmp22 = tmp21a + tmp60
    tmp23 = tmp22 + tmp63
    tmp24a = tmp23 + tmp66
    tmp25 = tmp24a + tmp69
    tmp26 = tmp25 + tmp72
    tmp27a = tmp26 + tmp75
    tmp28 = tmp27a + tmp78
    tmp29 = tmp28 + tmp81
    tmp30a = tmp29 + tmp84
    tmp31 = tmp30a + tmp87
    tmp32 = tmp31 + tmp90
    tmp33a = tmp32 + tmp93
    tmp34 = tmp33a + tmp96
    tmp35 = tmp34 + tmp99
    tmp36a = tmp35 + tmp102
    tmp37 = tmp36a + tmp105
    tmp38 = tmp37 + tmp108
    tmp39a = tmp38 + tmp111
    tmp40 = tmp39a + tmp114
    tmp41 = tmp40 + tmp117
    tmp42a = tmp41 + tmp0
    tmp43 = tl.broadcast_to(tmp42a, [XBLOCK])
    tmp44 = tl.sum(tmp43, 0)[:, None]
    tmp46 = tmp44 / 4096.0
    tmp47 = tmp0 - tmp46
    tmp49 = tmp47 * tmp47
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.sum(tmp50, 0)[:, None]
    tmp53 = 4096.0
    tmp55 = tmp52 / tmp53
    tmp56 = triton_helpers.maximum(tmp55, 1e-05)
    tmp58 = tl.sqrt(tmp55)
    tl.store(out_ptr0 + x0, tmp56, xmask)
    tl.store(out_ptr1 + x0, tmp58, xmask)


@triton.jit
def triton_poi_fused_add_mean_sqrt_3(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr0 + (4096 + x0), xmask)
    tmp4 = tl.load(in_ptr1 + (4096 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (8192 + x0), xmask)
    tmp8 = tl.load(in_ptr1 + (8192 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (12288 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (12288 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (16384 + x0), xmask)
    tmp16 = tl.load(in_ptr1 + (16384 + x0), xmask)
    tmp19 = tl.load(in_ptr0 + (20480 + x0), xmask)
    tmp20 = tl.load(in_ptr1 + (20480 + x0), xmask)
    tmp23 = tl.load(in_ptr0 + (24576 + x0), xmask)
    tmp24 = tl.load(in_ptr1 + (24576 + x0), xmask)
    tmp27 = tl.load(in_ptr0 + (28672 + x0), xmask)
    tmp28 = tl.load(in_ptr1 + (28672 + x0), xmask)
    tmp31 = tl.load(in_ptr0 + (32768 + x0), xmask)
    tmp32 = tl.load(in_ptr1 + (32768 + x0), xmask)
    tmp35 = tl.load(in_ptr0 + (36864 + x0), xmask)
    tmp36 = tl.load(in_ptr1 + (36864 + x0), xmask)
    tmp39 = tl.load(in_ptr0 + (40960 + x0), xmask)
    tmp40 = tl.load(in_ptr1 + (40960 + x0), xmask)
    tmp43 = tl.load(in_ptr0 + (45056 + x0), xmask)
    tmp44 = tl.load(in_ptr1 + (45056 + x0), xmask)
    tmp47 = tl.load(in_ptr0 + (49152 + x0), xmask)
    tmp48 = tl.load(in_ptr1 + (49152 + x0), xmask)
    tmp51 = tl.load(in_ptr0 + (53248 + x0), xmask)
    tmp52 = tl.load(in_ptr1 + (53248 + x0), xmask)
    tmp55 = tl.load(in_ptr0 + (57344 + x0), xmask)
    tmp56 = tl.load(in_ptr1 + (57344 + x0), xmask)
    tmp59 = tl.load(in_ptr0 + (61440 + x0), xmask)
    tmp60 = tl.load(in_ptr1 + (61440 + x0), xmask)
    tmp63 = tl.load(in_ptr0 + (65536 + x0), xmask)
    tmp64 = tl.load(in_ptr1 + (65536 + x0), xmask)
    tmp67 = tl.load(in_ptr0 + (69632 + x0), xmask)
    tmp68 = tl.load(in_ptr1 + (69632 + x0), xmask)
    tmp71 = tl.load(in_ptr0 + (73728 + x0), xmask)
    tmp72 = tl.load(in_ptr1 + (73728 + x0), xmask)
    tmp75 = tl.load(in_ptr0 + (77824 + x0), xmask)
    tmp76 = tl.load(in_ptr1 + (77824 + x0), xmask)
    tmp79 = tl.load(in_ptr0 + (81920 + x0), xmask)
    tmp80 = tl.load(in_ptr1 + (81920 + x0), xmask)
    tmp83 = tl.load(in_ptr0 + (86016 + x0), xmask)
    tmp84 = tl.load(in_ptr1 + (86016 + x0), xmask)
    tmp87 = tl.load(in_ptr0 + (90112 + x0), xmask)
    tmp88 = tl.load(in_ptr1 + (90112 + x0), xmask)
    tmp91 = tl.load(in_ptr0 + (94208 + x0), xmask)
    tmp92 = tl.load(in_ptr1 + (94208 + x0), xmask)
    tmp95 = tl.load(in_ptr0 + (98304 + x0), xmask)
    tmp96 = tl.load(in_ptr1 + (98304 + x0), xmask)
    tmp99 = tl.load(in_ptr0 + (102400 + x0), xmask)
    tmp100 = tl.load(in_ptr1 + (102400 + x0), xmask)
    tmp103 = tl.load(in_ptr0 + (106496 + x0), xmask)
    tmp104 = tl.load(in_ptr1 + (106496 + x0), xmask)
    tmp107 = tl.load(in_ptr0 + (110592 + x0), xmask)
    tmp108 = tl.load(in_ptr1 + (110592 + x0), xmask)
    tmp111 = tl.load(in_ptr0 + (114688 + x0), xmask)
    tmp112 = tl.load(in_ptr1 + (114688 + x0), xmask)
    tmp115 = tl.load(in_ptr0 + (118784 + x0), xmask)
    tmp116 = tl.load(in_ptr1 + (118784 + x0), xmask)
    tmp119 = tl.load(in_ptr0 + (122880 + x0), xmask)
    tmp120 = tl.load(in_ptr1 + (122880 + x0), xmask)
    tmp123 = tl.load(in_ptr0 + (126976 + x0), xmask)
    tmp124 = tl.load(in_ptr1 + (126976 + x0), xmask)
    tmp127 = tl.load(in_ptr0 + (131072 + x0), xmask)
    tmp128 = tl.load(in_ptr1 + (131072 + x0), xmask)
    tmp131 = tl.load(in_ptr0 + (135168 + x0), xmask)
    tmp132 = tl.load(in_ptr1 + (135168 + x0), xmask)
    tmp135 = tl.load(in_ptr0 + (139264 + x0), xmask)
    tmp136 = tl.load(in_ptr1 + (139264 + x0), xmask)
    tmp139 = tl.load(in_ptr0 + (143360 + x0), xmask)
    tmp140 = tl.load(in_ptr1 + (143360 + x0), xmask)
    tmp143 = tl.load(in_ptr0 + (147456 + x0), xmask)
    tmp144 = tl.load(in_ptr1 + (147456 + x0), xmask)
    tmp147 = tl.load(in_ptr0 + (151552 + x0), xmask)
    tmp148 = tl.load(in_ptr1 + (151552 + x0), xmask)
    tmp151 = tl.load(in_ptr0 + (155648 + x0), xmask)
    tmp152 = tl.load(in_ptr1 + (155648 + x0), xmask)
    tmp155 = tl.load(in_ptr0 + (159744 + x0), xmask)
    tmp156 = tl.load(in_ptr1 + (159744 + x0), xmask)
    tmp159 = tl.load(in_ptr0 + (163840 + x0), xmask)
    tmp160 = tl.load(in_ptr1 + (163840 + x0), xmask)
    tmp0_1 = tmp1 + tmp4
    tmp2_1 = tmp0_1 + tmp7
    tmp3_1 = tmp2_1 + tmp10
    tmp4_1 = tmp3_1 + tmp13
    tmp5_1 = tmp4_1 + tmp16
    tmp6_1 = tmp5_1 + tmp19
    tmp7_1 = tmp6_1 + tmp22
    tmp8_1 = tmp7_1 + tmp25
    tmp9_1 = tmp8_1 + tmp28
    tmp10_1 = tmp9_1 + tmp31
    tmp11_1 = tmp10_1 + tmp34
    tmp12_1 = tmp11_1 + tmp37
    tmp13_1 = tmp12_1 + tmp40
    tmp14_1 = tmp13_1 + tmp43
    tmp15_1 = tmp14_1 + tmp46
    tmp16_1 = tmp15_1 + tmp49
    tmp17_1 = tmp16_1 + tmp52
    tmp18_1 = tmp17_1 + tmp55
    tmp19_1 = tmp18_1 + tmp58
    tmp20_1 = tmp19_1 + tmp61
    tmp21_1 = tmp20_1 + tmp64
    tmp22_1 = tmp21_1 + tmp67
    tmp23_1 = tmp22_1 + tmp70
    tmp24_1 = tmp23_1 + tmp73
    tmp25_1 = tmp24_1 + tmp76
    tmp26_1 = tmp25_1 + tmp79
    tmp27_1 = tmp26_1 + tmp82
    tmp28_1 = tmp27_1 + tmp85
    tmp29_1 = tmp28_1 + tmp88
    tmp30_1 = tmp29_1 + tmp91
    tmp31_1 = tmp30_1 + tmp94
    tmp32_1 = tmp31_1 + tmp97
    tmp33_1 = tmp32_1 + tmp100
    tmp34_1 = tmp33_1 + tmp103
    tmp35_1 = tmp34_1 + tmp106
    tmp36_1 = tmp35_1 + tmp109
    tmp37_1 = tmp36_1 + tmp112
    tmp38_1 = tmp37_1 + tmp115
    tmp39_1 = tmp38_1 + tmp118
    tmp40_1 = tmp39_1 + tmp121
    tmp41_1 = tmp40_1 + tmp124
    tmp42_1 = tmp41_1 + tmp127
    tmp43_1 = tmp42_1 + tmp130
    tmp44_1 = tmp43_1 + tmp133
    tmp45_1 = tmp44_1 + tmp136
    tmp46_1 = tmp45_1 + tmp139
    tmp47_1 = tmp46_1 + tmp142
    tmp48_1 = tmp47_1 + tmp145
    tmp49_1 = tmp48_1 + tmp148
    tmp50_1 = tmp49_1 + tmp151
    tmp51_1 = tmp50_1 + tmp154
    tmp52_1 = tmp51_1 + tmp157
    tmp53_1 = tmp52_1 + tmp160
    tmp54 = tmp53_1 / 16384.0
    tmp55_1 = tmp1 - tmp54
    tmp56_1 = tmp55_1 * tmp55_1
    tmp57_1 = tmp56_1 + tmp26
    tmp58_1 = tmp57_1 + tmp29
    tmp59_1 = tmp58_1 + tmp32
    tmp60_1 = tmp59_1 + tmp35
    tmp61_1 = tmp60_1 + tmp38
    tmp62_1 = tmp61_1 + tmp41
    tmp63_1 = tmp62_1 + tmp44
    tmp64_1 = tmp63_1 + tmp47
    tmp65_1 = tmp64_1 + tmp50
    tmp66_1 = tmp65_1 + tmp53
    tmp67_1 = tmp66_1 + tmp56
    tmp68_1 = tmp67_1 + tmp59
    tmp69_1 = tmp68_1 + tmp62
    tmp70_1 = tmp69_1 + tmp65
    tmp71_1 = tmp70_1 + tmp68
    tmp72_1 = tmp71_1 + tmp71
    tmp73_1 = tmp72_1 + tmp74
    tmp74_1 = tmp73_1 + tmp77
    tmp75_1 = tmp74_1 + tmp80
    tmp76_1 = tmp75_1 + tmp83
    tmp77_1 = tmp76_1 + tmp86
    tmp78_1 = tmp77_1 + tmp89
    tmp79_1 = tmp78_1 + tmp92
    tmp80_1 = tmp79_1 + tmp95
    tmp81_1 = tmp80_1 + tmp98
    tmp82_1 = tmp81_1 + tmp101
    tmp83_1 = tmp82_1 + tmp104
    tmp84_1 = tmp83_1 + tmp107
    tmp85_1 = tmp84_1 + tmp110
    tmp86_1 = tmp85_1 + tmp113
    tmp87_1 = tmp86_1 + tmp116
    tmp88_1 = tmp87_1 + tmp119
    tmp89_1 = tmp88_1 + tmp122
    tmp90_1 = tmp89_1 + tmp125
    tmp91_1 = tmp90_1 + tmp128
    tmp92_1 = tmp91_1 + tmp131
    tmp93_1 = tmp92_1 + tmp134
    tmp94_1 = tmp93_1 + tmp137
    tmp95_1 = tmp94_1 + tmp140
    tmp96_1 = tmp95_1 + tmp143
    tmp97_1 = tmp96_1 + tmp146
    tmp98_1 = tmp97_1 + tmp149
    tmp99_1 = tmp98_1 + tmp152
    tmp100_1 = tmp99_1 + tmp155
    tmp101_1 = tmp100_1 + tmp158
    tmp102_1 = tmp101_1 + tmp161
    tmp103_1 = tmp102_1 / 16384.0
    tmp104_1 = tmp56_1 + tmp107
    tmp105_1 = tmp104_1 + tmp110
    tmp106_1 = tmp105_1 + tmp113
    tmp107_1 = tmp106_1 + tmp116
    tmp108_1 = tmp107_1 + tmp119
    tmp109_1 = tmp108_1 + tmp122
    tmp110_1 = tmp109_1 + tmp125
    tmp111_1 = tmp110_1 + tmp128
    tmp112_1 = tmp111_1 + tmp131
    tmp113_1 = tmp112_1 + tmp134
    tmp114_1 = tmp113_1 + tmp137
    tmp115_1 = tmp114_1 + tmp140
    tmp116_1 = tmp115_1 + tmp143
    tmp117_1 = tmp116_1 + tmp146
    tmp118_1 = tmp117_1 + tmp149
    tmp119_1 = tmp118_1 + tmp152
    tmp120_1 = tmp119_1 + tmp155
    tmp121_1 = tmp120_1 + tmp158
    tmp122_1 = tmp121_1 + tmp161
    tmp123_1 = tmp122_1 / 16384.0
    tmp124_1 = tmp56_1 + tmp127
    tmp125_1 = tmp124_1 + tmp130
    tmp126_1 = tmp125_1 + tmp133
    tmp127_1 = tmp126_1 + tmp136
    tmp128_1 = tmp127_1 + tmp139
    tmp129_1 = tmp128_1 + tmp142
    tmp130_1 = tmp129_1 + tmp145
    tmp131_1 = tmp130_1 + tmp148
    tmp132_1 = tmp131_1 + tmp151
    tmp133_1 = tmp132_1 + tmp154
    tmp134_1 = tmp133_1 + tmp157
    tmp135_1 = tmp134_1 + tmp160
    tmp136_1 = tmp135_1 / 16384.0
    tmp137_1 = tmp56_1 + tmp141
    tmp138_1 = tmp137_1 + tmp144
    tmp139_1 = tmp138_1 + tmp147
    tmp140_1 = tmp139_1 + tmp150
    tmp141_1 = tmp140_1 + tmp153
    tmp142_1 = tmp141_1 + tmp156
    tmp143_1 = tmp142_1 + tmp159
    tmp144_1 = tmp143_1 + tmp162
    tmp145_1 = tmp144_1 / 16384.0
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_mean_sqrt_4(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr0 + (4096 + x0), xmask)
    tmp4 = tl.load(in_ptr1 + (4096 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (8192 + x0), xmask)
    tmp8 = tl.load(in_ptr1 + (8192 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (12288 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (12288 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (16384 + x0), xmask)
    tmp16 = tl.load(in_ptr1 + (16384 + x0), xmask)
    tmp19 = tl.load(in_ptr0 + (20480 + x0), xmask)
    tmp20 = tl.load(in_ptr1 + (20480 + x0), xmask)
    tmp23 = tl.load(in_ptr0 + (24576 + x0), xmask)
    tmp24 = tl.load(in_ptr1 + (24576 + x0), xmask)
    tmp27 = tl.load(in_ptr0 + (28672 + x0), xmask)
    tmp28 = tl.load(in_ptr1 + (28672 + x0), xmask)
    tmp31 = tl.load(in_ptr0 + (32768 + x0), xmask)
    tmp32 = tl.load(in_ptr1 + (32768 + x0), xmask)
    tmp35 = tl.load(in_ptr0 + (36864 + x0), xmask)
    tmp36 = tl.load(in_ptr1 + (36864 + x0), xmask)
    tmp39 = tl.load(in_ptr0 + (40960 + x0), xmask)
    tmp40 = tl.load(in_ptr1 + (40960 + x0), xmask)
    tmp43 = tl.load(in_ptr0 + (45056 + x0), xmask)
    tmp44 = tl.load(in_ptr1 + (45056 + x0), xmask)
    tmp47 = tl.load(in_ptr0 + (49152 + x0), xmask)
    tmp48 = tl.load(in_ptr1 + (4915