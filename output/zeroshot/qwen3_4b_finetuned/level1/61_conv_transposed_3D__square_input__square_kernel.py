import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    constexpr):
    xnumel = 248832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 48
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 124416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 48
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (4096 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8192 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (12288 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16384 + x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (20480 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (24576 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (28672 + x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (32768 + x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (36864 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (40960 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (45056 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (49152 + x1), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (53248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (57344 + x1), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (61440 + x1), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (65536 + x1), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (69632 + x1), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (73728 + x1), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (77824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (81920 + x1), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (86016 + x1), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr0 + (90112 + x1), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (94208 + x1), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (98304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (102400 + x1), xmask, eviction_policy='evict_last'
        )
    tmp26 = tl.load(in_ptr0 + (106496 + x1), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (110592 + x1), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr0 + (114688 + x1), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (118784 + x1), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr0 + (122880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (126976 + x1), xmask, eviction_policy='evict_last'
        )
    tmp32 = tl.load(in_ptr0 + (131072 + x1), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (135168 + x1), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr0 + (139264 + x1), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (143360 + x1), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr0 + (147456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (151552 + x1), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr0 + (155648 + x1), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (159744 + x1), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr0 + (163840 + x1), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (167936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (172032 + x1), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (176128 + x1), xmask, eviction_policy='evict_last'
        )
    tmp44 = tl.load(in_ptr0 + (180224 + x1), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (184320 + x1), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr0 + (188416 + x1), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (192512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (196608 + x1), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (200704 + x1), xmask, eviction_policy='evict_last'
        )
    tmp50 = tl.load(in_ptr0 + (204800 + x1), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (208896 + x1), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr0 + (212992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (217088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (221184 + x1), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (225280 + x1), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr0 + (229376 + x1), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (233472 + x1), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr0 + (237568 + x1), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (241664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (245760 + x1), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (249856 + x1), xmask, eviction_policy='evict_last'
        )
    tmp62 = tl.load(in_ptr0 + (253952 + x1), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (258048 + x1), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr0 + (262144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (266240 + x1), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (270336 + x1), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (274432 + x1), xmask, eviction_policy='evict_last'
        )
    tmp68 = tl.load(in_ptr0 + (278528 + x1), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (282624 + x1), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr0 + (286720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (290816 + x1), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (294912 + x1), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (299008 + x1), xmask, eviction_policy='evict_last'
        )
    tmp74 = tl.load(in_ptr0 + (303104 + x1), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (307200 + x1), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr0 + (311296 + x1), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (315392 + x1), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (319488 + x1), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (323584 + x1), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr0 + (327680 + x1), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (331776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr0 + (335872 + x1), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (339968 + x1), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (344064 + x1), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (348160 + x1), xmask, eviction_policy='evict_last'
        )
    tmp86 = tl.load(in_ptr0 + (352256 + x1), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (356352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr0 + (360448 + x1), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (364544 + x1), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (368640 + x1), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (372736 + x1), xmask, eviction_policy='evict_last'
        )
    tmp92 = tl.load(in_ptr0 + (376832 + x1), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (380928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr0 + (385024 + x1), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (389120 + x1), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (393216 + x1), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (397312 + x1), xmask, eviction_policy='evict_last'
        )
    tmp98 = tl.load(in_ptr0 + (401408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (405504 + x1), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr0 + (409600 + x1), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (413696 + x1), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (417792 + x1), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (421888 + x1), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr0 + (425984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (430080 + x1), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr0 + (434176 + x1), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (438272 + x1), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (442368 + x1), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (446464 + x1), xmask, eviction_policy='evict_last'
        )
    tmp110 = tl.load(in_ptr0 + (450560 + x1), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (454656 + x1), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr0 + (458752 + x1), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (462848 + x1), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (466944 + x1), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (471040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp116 = tl.load(in_ptr0 + (475136 + x1), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (479232 + x1), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr0 + (483328 + x1), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (487424 + x1), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (491520 + x1), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (495616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp122 = tl.load(in_ptr0 + (499712 + x1), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (503808 + x1), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr0 + (507904 + x1), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr0 + (511999 + x1), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (516096 + x1), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (520192 + x1), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr0 + (524288 + x1), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (528384 + x1), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr0 + (532480 + x1), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr0 + (536576 + x1), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (540672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (544768 + x1), xmask, eviction_policy='evict_last'
        )
    tmp134 = tl.load(in_ptr0 + (548864 + x1), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (552960 + x1), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr0 + (557056 + x1), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr0 + (561152 + x1), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (565248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (569344 + x1), xmask, eviction_policy='evict_last'
        )
    tmp140 = tl.load(in_ptr0 + (573440 + x1), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (577536 + x1), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr0 + (581632 + x1), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr0 + (585728 + x1), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (589824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (593920 + x1), xmask, eviction_policy='evict_last'
        )
    tmp146 = tl.load(in_ptr0 + (598016 + x1), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (602112 + x1), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr0 + (606208 + x1), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr0 + (610304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (614400 + x1), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (618496 + x1), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr0 + (622592 + x1), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (626688 + x1), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr0 + (630784 + x1), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr0 + (634880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (638976 + x1), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (643072 + x1), xmask, eviction_policy='evict_last'
        )
    tmp158 = tl.load(in_ptr0 + (647168 + x1), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (651264 + x1), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr0 + (655360 + x1), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr0 + (659456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (663552 + x1), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (667648 + x1), xmask, eviction_policy='evict_last'
        )
    tmp164 = tl.load(in_ptr0 + (671744 + x1), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (675840 + x1), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr0 + (679936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr0 + (684032 + x1), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (688128 + x1), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (692224 + x1), xmask, eviction_policy='evict_last'
        )
    tmp170 = tl.load(in_ptr0 + (696320 + x1), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (700416 + x1), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr0 + (704512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr0 + (708608 + x1), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (712704 + x1), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (716800 + x1), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr0 + (720896 + x1), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (724992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr0 + (729088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr0 + (733184 + x1), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (737280 + x1), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (741376 + x1), xmask, eviction_policy='evict_last'
        )
    tmp182 = tl.load(in_ptr0 + (745472 + x1), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (749568 + x1), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr0 + (753664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr0 + (757760 + x1), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (761856 + x1), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (765952 + x1), xmask, eviction_policy='evict_last'
        )
    tmp188 = tl.load(in_ptr0 + (769948 + x1), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (774044 + x1), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr0 + (778140 + x1), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr0 + (782236 + x1), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (786332 + x1), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (790428 + x1), xmask, eviction_policy='evict_last'
        )
    tmp194 = tl.load(in_ptr0 + (794524 + x1), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (798620 + x1), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr0 + (802716 + x1), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr0 + (806812 + x1), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (810908 + x1), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (814994 + x1), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr0 + (819090 + x1), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (823186 + x1), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr0 + (827282 + x1), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr0 + (831378 + x1), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (835474 + x1), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (839570 + x1), xmask, eviction_policy='evict_last'
        )
    tmp206 = tl.load(in_ptr0 + (843666 + x1), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (847762 + x1), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr0 + (851858 + x1), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr0 + (855954 + x1), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (859950 + x1), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (863946 + x1), xmask, eviction_policy='evict_last'
        )
    tmp212 = tl.load(in_ptr0 + (867942 + x1), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (871938 + x1), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl.load(in_ptr0 + (875934 + x1), xmask, eviction_policy='evict_last'
        )
    tmp215 = tl.load(in_ptr0 + (879930 + x1), xmask, eviction_policy='evict_last'
        )
    tmp216 = tl.load(in_ptr0 + (883926 + x1), xmask, eviction_policy='evict_last'
        )
    tmp217 = tl.load(in_ptr0 + (887922 + x1), xmask, eviction_policy='evict_last'
        )
    tmp218 = tl.load(in_ptr0 + (891918 + x1), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (895914 + x1), xmask, eviction_policy='evict_last'
        )
    tmp220 = tl.load(in_ptr0 + (899910 + x1), xmask, eviction_policy='evict_last'
        )
    tmp221 = tl.load(in_ptr0 + (903906 + x1), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr0 + (907902 + x1), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr0 + (911898 + x1), xmask, eviction_policy='evict_last'
        )
    tmp224 = tl.load(in_ptr0 + (915894 + x1), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (919890 + x1), xmask, eviction_policy='evict_last'
        )
    tmp226 = tl.load(in_ptr0 + (923886 + x1), xmask, eviction_policy='evict_last'
        )
    tmp227 = tl.load(in_ptr0 + (927882 + x1), xmask, eviction_policy='evict_last'
        )
   