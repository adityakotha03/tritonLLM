import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_diag_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_index_put_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4096 % 4096
    x0 = xindex % 4096
    x2 = xindex // 16777216
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (x1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp214 =