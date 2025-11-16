import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 15865024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = -tmp0
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = triton_helpers.maximum(tmp2, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_div_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15865024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 393216
    x1 = xindex // 393216
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (393216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr1 + (786432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr1 + (1179648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr1 + (1572864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr1 + (1966080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr1 + (2359296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr1 + (2752512 + x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr1 + (3145728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr1 + (3538944 + x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr1 + (3932160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr1 + (4325376 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr1 + (4718592 + x0), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr1 + (5111808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr1 + (5505024 + x0), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr1 + (5898240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr1 + (6291456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr1 + (6684672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr1 + (7077888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr1 + (7471104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr1 + (7864320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr1 + (8257536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr1 + (8650752 + x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr1 + (9043968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr1 + (9437184 + x0), xmask, eviction_policy='evict_last'
        )
    tmp26 = tl.load(in_ptr1 + (9830400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr1 + (10223616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr1 + (10616832 + x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr1 + (11009952 + x0), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr1 + (11403168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr1 + (11796384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp32 = tl.load(in_ptr1 + (12189600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr1 + (12582816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr1 + (12976032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr1 + (13369248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr1 + (13762464 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr1 + (14155680 + x0), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr1 + (14548896 + x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr1 + (14942112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr1 + (15335328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr1 + (15728544 + x0), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr1 + (16121760 + x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr1 + (16514976 + x0), xmask, eviction_policy='evict_last'
        )
    tmp44 = tl.load(in_ptr1 + (16908192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr1 + (17301408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr1 + (17694624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr1 + (18087840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr1 + (18481056 + x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr1 + (18874272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp50 = tl.load(in_ptr1 + (19267488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr1 + (19660704 + x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr1 + (20053920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr1 + (20447136 + x0), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr1 + (20840352 + x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr1 + (21233568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr1 + (21626784 + x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr1 + (22019999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr1 + (22413215 + x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr1 + (22806431 + x0), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr1 + (23199647 + x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr1 + (23592863 + x0), xmask, eviction_policy='evict_last'
        )
    tmp62 = tl.load(in_ptr1 + (23986079 + x0), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr1 + (24379295 + x0), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr1 + (24772511 + x0), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr1 + (25165727 + x0), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr1 + (25558943 + x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr1 + (25952159 + x0), xmask, eviction_policy='evict_last'
        )
    tmp68 = tl.load(in_ptr1 + (26345375 + x0), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr1 + (26738591 + x0), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr1 + (27131807 + x0), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr1 + (27525023 + x0), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr1 + (27918239 + x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr1 + (28311455 + x0), xmask, eviction_policy='evict_last'
        )
    tmp74 = tl.load(in_ptr1 + (28704671 + x0), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr1 + (29097887 + x0), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr1 + (29491103 + x0), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr1 + (29884319 + x0), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr1 + (30277535 + x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr1 + (30670751 + x0), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr1 + (31063967 + x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr1 + (31457183 + x0), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr1 + (31850399 + x0), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr1 + (32243615 + x0), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr1 + (32636831 + x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr1 + (33029947 + x0), xmask, eviction_policy='evict_last'
        )
    tmp86 = tl.load(in_ptr1 + (33423163 + x0), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr1 + (33816379 + x0), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr1 + (34209595 + x0), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr1 + (34602811 + x0), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr1 + (34996027 + x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr1 + (35389243 + x0), xmask, eviction_policy='evict_last'
        )
    tmp92 = tl.load(in_ptr1 + (35782459 + x0), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr1 + (36175675 + x0), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr1 + (36568891 + x0), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr1 + (36962107 + x0), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr1 + (37355323 + x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr1 + (37748539 + x0), xmask, eviction_policy='evict_last'
        )
    tmp98 = tl.load(in_ptr1 + (38141755 + x0), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr1 + (38534971 + x0), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr1 + (38928187 + x0), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr1 + (39321403 + x0), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr1 + (39714619 + x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr1 + (40107835 + x0), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr1 + (40501051 + x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr1 + (40894267 + x0), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr1 + (41287483 + x0), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr1 + (41680699 + x0), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr1 + (42073915 + x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr1 + (42467131 + x0), xmask, eviction_policy='evict_last'
        )
    tmp110 = tl.load(in_ptr1 + (42860347 + x0), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr1 + (43253563 + x0), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr1 + (43646779 + x0), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr1 + (44040000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr1 + (44433216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr1 + (44826432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp116 = tl.load(in_ptr1 + (45219648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr1 + (45612864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr1 + (46006080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr1 + (46399296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr1 + (46792512 + x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr1 + (47185728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp122 = tl.load(in_ptr1 + (47578944 + x0), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr1 + (47972160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr1 + (48365376 + x0), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr1 + (48758592 + x0), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr1 + (49151808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr1 + (49545024 + x0), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr1 + (49938240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr1 + (50331456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr1 + (50724672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr1 + (51117888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr1 + (51511104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr1 + (51904320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp134 = tl.load(in_ptr1 + (52297536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr1 + (52690752 + x0), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr1 + (53083968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr1 + (53477184 + x0), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr1 + (53870400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr1 + (54263616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp140 = tl.load(in_ptr1 + (54656832 + x0), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr1 + (55050048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr1 + (55443264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr1 + (55836480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr1 + (56229696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr1 + (56622912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp146 = tl.load(in_ptr1 + (57016128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr1 + (57409344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr1 + (57802560 + x0), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr1 + (58195776 + x0), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr1 + (58588992 + x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr1 + (58982208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr1 + (59375424 + x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr1 + (59768640 + x0), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr1 + (60161856 + x0), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr1 + (60555072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr1 + (60948288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr1 + (61341504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp158 = tl.load(in_ptr1 + (61734720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr1 + (62127936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr1 + (62521152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr1 + (62914368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr1 + (63307584 + x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr1 + (63700800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp164 = tl.load(in_ptr1 + (64094016 + x0), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr1 + (64487232 + x0), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr1 + (64880448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr1 + (65273664 + x0), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr1 + (65666880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr1 + (66060096 + x0), xmask, eviction_policy='evict_last'
        )
    tmp170 = tl.load(in_ptr1 + (66453312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr1 + (66846528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr1 + (67239744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr1 + (67632960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr1 + (68026176 + x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr1 + (68419392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr1 + (68812608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr1 + (69205824 + x0), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr1 + (69599040 + x0), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr1 + (69992256 + x0), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr1 + (70385472 + x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr1 + (70778688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp182 = tl.load(in_ptr1 + (71171904 + x0), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr1 + (71565120 + x0), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr1 + (71958336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr1 + (72351552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr1 + (72744768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr1 + (73137984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp188 = tl.load(in_ptr1 + (73531200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr1 + (73924416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr1 + (74317632 + x0), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr1 + (74710848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr1 + (75104064 + x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr1 + (75497280 + x0), xmask, eviction_policy='evict_last'
        )
    tmp194 = tl.load(in_ptr1 + (75890496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr1 + (76283712 + x0), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr1 + (76676928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr1 + (77070144 + x0), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr1 + (77463360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr1 + (77856576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr1 + (78249792 + x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr1 + (78643008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr1 + (79036224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr1 + (79429440 + x0), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr1 + (79822656 + x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr1 + (80215872 + x0), xmask, eviction_policy='evict_last'
        )
    tmp206 = tl.load(in_ptr1 + (80609088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr1 + (80992304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr1 + (81385520 + x0), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr1 + (81778736 + x0), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr1 + (82171952 + x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr1 + (82565168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp212 = tl.load(in_ptr1 + (82958384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr1 + (83351600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl