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
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp1 + tmp3
    tmp4 = tmp2 + tmp5
    tmp6 = tmp4 + tmp7
    tmp8 = tmp6 + tmp9
    tmp10 = tmp8 + tmp11
    tmp12 = tmp10 + tmp13
    tmp14 = tmp12 + tmp15
    tmp16 = tmp14 + tmp17
    tmp18 = tmp16 + tmp19
    tmp20 = tmp18 + tmp21
    tmp22 = tmp20 + tmp23
    tmp24 = tmp22 + tmp25
    tmp26 = tmp24 + tmp27
    tmp28 = tmp26 + tmp29
    tmp30 = tmp28 + tmp31
    tmp32 = tmp30 + tmp33
    tmp34 = tmp32 + tmp35
    tmp36 = tmp34 + tmp37
    tmp38 = tmp36 + tmp39
    tmp40 = tmp38 + tmp41
    tmp42 = tmp40 + tmp43
    tmp44 = tmp42 + tmp45
    tmp46 = tmp44 + tmp47
    tmp48 = tmp46 + tmp49
    tmp50 = tmp48 + tmp51
    tmp52 = tmp50 + tmp53
    tmp54 = tmp52 + tmp55
    tmp56 = tmp54 + tmp57
    tmp58 = tmp56 + tmp59
    tmp60 = tmp58 + tmp61
    tmp62 = tmp0 + tmp2
    tmp63 = tmp62 + tmp4
    tmp64 = tmp63 + tmp6
    tmp65 = tmp64 + tmp8
    tmp66 = tmp65 + tmp10
    tmp67 = tmp66 + tmp12
    tmp68 = tmp67 + tmp14
    tmp69 = tmp68 + tmp16
    tmp70 = tmp69 + tmp18
    tmp71 = tmp70 + tmp20
    tmp72 = tmp71 + tmp22
    tmp73 = tmp72 + tmp24
    tmp74 = tmp73 + tmp26
    tmp75 = tmp74 + tmp28
    tmp76 = tmp75 + tmp30
    tmp77 = tmp76 + tmp32
    tmp78 = tmp77 + tmp34
    tmp79 = tmp78 + tmp36
    tmp80 = tmp79 + tmp38
    tmp81 = tmp80 + tmp40
    tmp82 = tmp81 + tmp42
    tmp83 = tmp82 + tmp44
    tmp84 = tmp83 + tmp46
    tmp85 = tmp84 + tmp48
    tmp86 = tmp85 + tmp50
    tmp87 = tmp86 + tmp52
    tmp88 = tmp87 + tmp54
    tmp89 = tmp88 + tmp56
    tmp90 = tmp89 + tmp58
    tmp91 = tmp90 + tmp60
    tmp92 = tmp62 * tmp62
    tmp93 = tmp64 * tmp64
    tmp94 = tmp65 * tmp65
    tmp95 = tmp66 * tmp66
    tmp96 = tmp67 * tmp67
    tmp97 = tmp68 * tmp68
    tmp98 = tmp69 * tmp69
    tmp99 = tmp70 * tmp70
    tmp100 = tmp71 * tmp71
    tmp101 = tmp72 * tmp72
    tmp102 = tmp73 * tmp73
    tmp103 = tmp74 * tmp74
    tmp104 = tmp75 * tmp75
    tmp105 = tmp76 * tmp76
    tmp106 = tmp77 * tmp77
    tmp107 = tmp78 * tmp78
    tmp108 = tmp79 * tmp79
    tmp109 = tmp80 * tmp80
    tmp110 = tmp81 * tmp81
    tmp111 = tmp82 * tmp82
    tmp112 = tmp83 * tmp83
    tmp113 = tmp84 * tmp84
    tmp114 = tmp85 * tmp85
    tmp115 = tmp86 * tmp86
    tmp116 = tmp87 * tmp87
    tmp117 = tmp88 * tmp88
    tmp118 = tmp89 * tmp89
    tmp119 = tmp90 * tmp90
    tmp120 = tmp91 * tmp91
    tmp121 = tmp92 * tmp92
    tmp122 = tmp93 + tmp95
    tmp123 = tmp122 + tmp97
    tmp124 = tmp123 + tmp99
    tmp125 = tmp124 + tmp101
    tmp126 = tmp125 + tmp103
    tmp127 = tmp126 + tmp105
    tmp128 = tmp127 + tmp107
    tmp129 = tmp128 + tmp109
    tmp130 = tmp129 + tmp111
    tmp131 = tmp130 + tmp113
    tmp132 = tmp131 + tmp115
    tmp133 = tmp132 + tmp117
    tmp134 = tmp133 + tmp119
    tmp135 = tmp134 + tmp121
    tmp136 = tmp135 / 16.0
    tmp137 = 0.0
    tmp138 = tmp137 - tmp136
    tmp139 = 1e-05
    tmp140 = tmp139 + tmp138
    tmp141 = triton_helpers.maximum(tmp140, 0.0)
    tl.store(out_ptr0 + x2, tmp141, xmask)
    tl.store(out_ptr1 + x2, tmp136, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp1 + tmp3
    tmp4 = tmp2 + tmp5
    tmp6 = tmp4 + tmp7
    tmp8 = tmp6 + tmp9
    tmp10 = tmp8 + tmp11
    tmp12 = tmp10 + tmp13
    tmp14 = tmp12 + tmp15
    tmp16 = tmp14 + tmp17
    tmp18 = tmp16 + tmp19
    tmp20 = tmp18 + tmp21
    tmp22 = tmp20 + tmp23
    tmp24 = tmp22 + tmp25
    tmp26 = tmp24 + tmp27
    tmp28 = tmp26 + tmp29
    tmp30 = tmp28 + tmp31
    tmp32 = tmp30 + tmp33
    tmp34 = tmp32 + tmp35
    tmp36 = tmp34 + tmp37
    tmp38 = tmp36 + tmp39
    tmp40 = tmp38 + tmp41
    tmp42 = tmp40 + tmp43
    tmp44 = tmp42 + tmp45
    tmp46 = tmp44 + tmp47
    tmp48 = tmp46 + tmp49
    tmp50 = tmp48 + tmp51
    tmp52 = tmp50 + tmp53
    tmp54 = tmp52 + tmp55
    tmp56 = tmp54 + tmp57
    tmp58 = tmp56 + tmp59
    tmp60 = tmp58 + tmp61
    tmp62 = tmp0 + tmp2
    tmp63 = tmp62 + tmp4
    tmp64 = tmp63 + tmp6
    tmp65 = tmp64 + tmp8
    tmp66 = tmp65 + tmp10
    tmp67 = tmp66 + tmp12
    tmp68 = tmp67 + tmp14
    tmp69 = tmp68 + tmp16
    tmp70 = tmp69 + tmp18
    tmp71 = tmp70 + tmp20
    tmp72 = tmp71 + tmp22
    tmp73 = tmp72 + tmp24
    tmp74 = tmp73 + tmp26
    tmp75 = tmp74 + tmp28
    tmp76 = tmp75 + tmp30
    tmp77 = tmp76 + tmp32
    tmp78 = tmp77 + tmp34
    tmp79 = tmp78 + tmp36
    tmp80 = tmp79 + tmp38
    tmp81 = tmp80 + tmp40
    tmp82 = tmp81 + tmp42
    tmp83 = tmp82 + tmp44
    tmp84 = tmp83 + tmp46
    tmp85 = tmp84 + tmp48
    tmp86 = tmp85 + tmp50
    tmp87 = tmp86 + tmp52
    tmp88 = tmp87 + tmp54
    tmp89 = tmp88 + tmp56
    tmp90 = tmp89 + tmp58
    tmp91 = tmp90 + tmp60
    tmp92 = tmp62 * tmp62
    tmp93 = tmp64 * tmp64
    tmp94 = tmp65 * tmp65
    tmp95 = tmp66 * tmp66
    tmp96 = tmp67 * tmp67
    tmp97 = tmp68 * tmp68
    tmp98 = tmp69 * tmp69
    tmp99 = tmp70 * tmp70
    tmp100 = tmp71 * tmp71
    tmp101 = tmp72 * tmp72
    tmp102 = tmp73 * tmp73
    tmp103 = tmp74 * tmp74
    tmp104 = tmp75 * tmp75
    tmp105 = tmp76 * tmp76
    tmp106 = tmp77 * tmp77
    tmp107 = tmp78 * tmp78
    tmp108 = tmp79 * tmp79
    tmp109 = tmp80 * tmp80
    tmp110 = tmp81 * tmp81
    tmp111 = tmp82 * tmp82
    tmp112 = tmp83 * tmp83
    tmp113 = tmp84 * tmp84
    tmp114 = tmp85 * tmp85
    tmp115 = tmp86 * tmp86
    tmp116 = tmp87 * tmp87
    tmp117 = tmp88 * tmp88
    tmp118 = tmp89 * tmp89
    tmp119 = tmp90 * tmp90
    tmp120 = tmp91 * tmp91
    tmp121 = tmp92 * tmp92
    tmp122 = tmp93 + tmp95
    tmp123 = tmp122 + tmp97
    tmp124 = tmp123 + tmp99
    tmp125 = tmp124 + tmp101
    tmp126 = tmp125 + tmp103
    tmp127 = tmp126 + tmp105
    tmp128 = tmp127 + tmp107
    tmp129 = tmp128 + tmp109
    tmp130 = tmp129 + tmp111
    tmp131 = tmp130 + tmp113
    tmp132 = tmp131 + tmp115
    tmp133 = tmp132 + tmp117
    tmp134 = tmp133 + tmp119
    tmp135 = tmp134 + tmp121
    tmp136 = tmp135 / 16.0
    tmp137 = 0.0
    tmp138 = tmp137 - tmp136
    tmp139 = 1e-05
    tmp140 = tmp139 + tmp138
    tmp141 = triton_helpers.maximum(tmp140, 0.0)
    tl.store(out_ptr0 + x2, tmp141, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp1 + tmp3
    tmp4 = tmp2 + tmp5
    tmp6 = tmp4 + tmp7
    tmp8 = tmp6 + tmp9
    tmp10 = tmp8 + tmp11
    tmp12 = tmp10 + tmp13
    tmp14 = tmp12 + tmp15
    tmp16 = tmp14 + tmp17
    tmp18 = tmp16 + tmp19
    tmp20 = tmp18 + tmp21
    tmp22 = tmp20 + tmp23
    tmp24 = tmp22 + tmp25
    tmp26 = tmp24 + tmp27
    tmp28 = tmp26 + tmp29
    tmp30 = tmp28 + tmp31
    tmp32 = tmp30 + tmp33
    tmp34 = tmp32 + tmp35
    tmp36 = tmp34 + tmp37
    tmp38 = tmp36 + tmp39
    tmp40 = tmp38 + tmp41
    tmp42 = tmp40 + tmp43
    tmp44 = tmp42 + tmp45
    tmp46 = tmp44 + tmp47
    tmp48 = tmp46 + tmp49
    tmp50 = tmp48 + tmp51
    tmp52 = tmp50 + tmp53
    tmp54 = tmp52 + tmp55
    tmp56 = tmp54 + tmp57
    tmp58 = tmp56 + tmp59
    tmp60 = tmp58 + tmp61
    tmp62 = tmp0 + tmp2
    tmp63 = tmp62 + tmp4
    tmp64 = tmp63 + tmp6
    tmp65 = tmp64 + tmp8
    tmp66 = tmp65 + tmp10
    tmp67 = tmp66 + tmp12
    tmp68 = tmp67 + tmp14
    tmp69 = tmp68 + tmp16
    tmp70 = tmp69 + tmp18
    tmp71 = tmp70 + tmp20
    tmp72 = tmp71 + tmp22
    tmp73 = tmp72 + tmp24
    tmp74 = tmp73 + tmp26
    tmp75 = tmp74 + tmp28
    tmp76 = tmp75 + tmp30
    tmp77 = tmp76 + tmp32
    tmp78 = tmp77 + tmp34
    tmp79 = tmp78 + tmp36
    tmp80 = tmp79 + tmp38
    tmp81 = tmp80 + tmp40
    tmp82 = tmp81 + tmp42
    tmp83 = tmp82 + tmp44
    tmp84 = tmp83 + tmp46
    tmp85 = tmp84 + tmp48
    tmp86 = tmp85 + tmp50
    tmp87 = tmp86 + tmp52
    tmp88 = tmp87 + tmp54
    tmp89 = tmp88 + tmp56
    tmp90 = tmp89 + tmp58
    tmp91 = tmp90 + tmp60
    tmp92 = tmp62 * tmp62
    tmp93 = tmp64 * tmp64
    tmp94 = tmp65 * tmp65
    tmp95 = tmp66 * tmp66
    tmp96 = tmp67 * tmp67
    tmp97 = tmp68 * tmp68
    tmp98 = tmp69 * tmp69
    tmp99 = tmp70 * tmp70
    tmp100 = tmp71 * tmp71
    tmp101 = tmp72 * tmp72
    tmp102 = tmp73 * tmp73
    tmp103 = tmp74 * tmp74
    tmp104 = tmp75 * tmp75
    tmp105 = tmp76 * tmp76
    tmp106 = tmp77 * tmp77
    tmp107 = tmp78 * tmp78
    tmp108 = tmp79 * tmp79
    tmp109 = tmp80 * tmp80
    tmp110 = tmp81 * tmp81
    tmp111 = tmp82 * tmp82
    tmp112 = tmp83 * tmp83
    tmp113 = tmp84 * tmp84
    tmp114 = tmp85 * tmp85
    tmp115 = tmp86 * tmp86
    tmp116 = tmp87 * tmp87
    tmp117 = tmp88 * tmp88
    tmp118 = tmp89 * tmp89
    tmp119 = tmp90 * tmp90
    tmp120 = tmp91 * tmp91
    tmp121 = tmp92 * tmp92
    tmp122 = tmp93 + tmp95
    tmp123 = tmp122 + tmp97
    tmp124 = tmp123 + tmp99
    tmp125 = tmp124 + tmp101
    tmp126 = tmp125 + tmp103
    tmp127 = tmp126 + tmp105
    tmp128 = tmp127 + tmp107
    tmp129 = tmp128 + tmp109
    tmp130 = tmp129 + tmp111
    tmp131 = tmp130 + tmp113
    tmp132 = tmp131 + tmp115
    tmp133 = tmp132 + tmp117
    tmp134 = tmp133 + tmp119
    tmp135 = tmp134 + tmp121
    tmp136 = tmp135 / 16.0
    tmp137 = 0.0
    tmp138 = tmp137 - tmp136
    tmp139 = 1e-05
    tmp140 = tmp139 + tmp138
    tmp141 = triton_helpers.maximum(tmp140, 0.0)
    tl.store(out_ptr0 + x2, tmp141, xmask)


@triton.jit
def triton_poi_fused__native