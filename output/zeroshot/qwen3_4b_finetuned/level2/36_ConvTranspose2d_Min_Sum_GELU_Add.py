import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 128
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 131072), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1024 + x1 + 131072), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp5 = tmp4 + tmp1
    tmp7 = tmp6 + tmp5
    tl.store(out_ptr0 + x3, tmp7, xmask)


@triton.jit
def triton_poi_fused_minimum_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (256 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (384 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (640 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (768 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (928 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1088 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1248 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1408 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (1568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr0 + (1728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (1888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (2048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (2208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (2368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (2528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (2688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (2848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (3008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (3168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr0 + (3328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (3488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (3648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (3808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp26 = tl.load(in_ptr0 + (3968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (4128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr0 + (4288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (4448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr0 + (4608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (4768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp32 = tl.load(in_ptr0 + (4928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (5088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr0 + (5248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (5408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr0 + (5568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (5728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr0 + (5888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (6048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr0 + (6208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (6368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (6528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (6688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp44 = tl.load(in_ptr0 + (6848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (7008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr0 + (7168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (7328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (7488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (7648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp50 = tl.load(in_ptr0 + (7808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (7968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr0 + (8128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (8288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (8448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (8608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr0 + (8768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (8928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr0 + (9088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (9248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (9408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (9568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp62 = tl.load(in_ptr0 + (9728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (9888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr0 + (10048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (10208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (10368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (10528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp68 = tl.load(in_ptr0 + (10688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (10848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr0 + (11008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (11168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (11328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (11488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp74 = tl.load(in_ptr0 + (11648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (11808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr0 + (11968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (12128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (12288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (12448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr0 + (12608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (12768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr0 + (12928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (13088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (13248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (13408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp86 = tl.load(in_ptr0 + (13568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (13728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr0 + (13888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (14048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (14208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (14368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp92 = tl.load(in_ptr0 + (14528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (14688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr0 + (14848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (15008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (15168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (15328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp98 = tl.load(in_ptr0 + (15488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (15648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr0 + (15808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (15968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (16128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (16288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr0 + (16448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (16608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr0 + (16768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (16928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (17088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (17248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp110 = tl.load(in_ptr0 + (17408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (17568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr0 + (17728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (17888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (18048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (18208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp116 = tl.load(in_ptr0 + (18368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (18528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr0 + (18688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (18848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (19008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (19168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp122 = tl.load(in_ptr0 + (19328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (19488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr0 + (19648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr0 + (19808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (19968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (20128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr0 + (20288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (20448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr0 + (20608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr0 + (20768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (20928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (21088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp134 = tl.load(in_ptr0 + (21248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (21408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr0 + (21568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr0 + (21728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (21888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (22048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp140 = tl.load(in_ptr0 + (22208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (22368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr0 + (22528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr0 + (22688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (22848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (23008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp146 = tl.load(in_ptr0 + (23168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (23328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr0 + (23488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr0 + (23648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (23808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (23968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr0 + (24128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (24288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr0 + (24448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr0 + (24608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (24768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (24928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp158 = tl.load(in_ptr0 + (25088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (25248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr0 + (25408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr0 + (25568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (25728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (25888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp164 = tl.load(in_ptr0 + (26048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (26208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr0 + (26368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr0 + (26528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (26688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (26848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp170 = tl.load(in_ptr0 + (27008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (27168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr0 + (27328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr0 + (27488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (27648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (27808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr0 + (27968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (28128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr0 + (28288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr0 + (28448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (28608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (28768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp182 = tl.load(in_ptr0 + (28928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (29088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr0 + (29248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr0 + (29408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (29568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (29728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp188 = tl.load(in_ptr0 + (29888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (30048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr0 + (30208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr0 + (30368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (30528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (30688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp194 = tl.load(in_ptr0 + (30848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (31008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr0 + (31168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr0 + (31328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (31488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (31648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr0 + (31808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (31968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr0 + (32128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr0 + (32288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (32448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (32608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp206 = tl.load(in_ptr0 + (32768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (32928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr0 + (33088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr0 + (33248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (33408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (33568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp212 = tl.load(in_ptr0 + (33728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (33888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl.load(in_ptr0 + (34048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp215 = tl.load(in_ptr0 + (34208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp216 = tl.load(in_ptr0 + (34368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp217 = tl.load(in_ptr0 + (34528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp218 = tl.load(in_ptr0 + (34688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (34848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp220 = tl.load(in_ptr0 + (35008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp221 = tl.load(in_ptr0 + (35168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr0 + (35328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr0 + (35488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp224 = tl.load(in_ptr0 + (35648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (35808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp226 = tl.load(in_ptr0 + (35968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp227 = tl.load(in_ptr0 + (36128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp228 = tl.load(in_ptr0 + (