import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16496 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x2 = xindex // 262144
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 262144
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 16 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 16 * x2), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (262144 + 2 * x0 + 16 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (262145 + 2 * x0 + 16 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp2 == tmp0
    tmp8 = tmp1 == tmp0
    tmp9 = tmp7 | tmp8
    tmp10 = tmp3 == tmp0
    tmp11 = tmp5 == tmp0
    tmp12 = tmp10 | tmp11
    tmp13 = tmp9 | tmp12
    tmp14 = tmp4 == tmp2
    tmp15 = tmp3 == tmp2
    tmp16 = tmp14 | tmp15
    tmp17 = tmp5 == tmp2
    tmp18 = tmp16 | tmp17
    tmp19 = tmp13 | tmp18
    tmp20 = tmp5 == tmp4
    tmp21 = tmp4 == tmp2
    tmp22 = tmp20 | tmp21
    tmp23 = tmp19 | tmp22
    tmp24 = tmp6 == tmp4
    tmp25 = tmp5 == tmp4
    tmp26 = tmp24 | tmp25
    tmp27 = tmp23 | tmp26
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp27, xmask)


@triton.jit
def triton_poi_fused_global_avg_pool3d_3(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.sum(tmp1, 0)[:, None]
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_clamp_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_out_ptr0 + (16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_out_ptr0 + (16 * x0 + 16), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_out_ptr0 + (16 * x0 + 1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_out_ptr0 + (16 * x0 + 17), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_out_ptr0 + (16 * x0 + 8), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_out_ptr0 + (16 * x0 + 9), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_out_ptr0 + (16 * x0 + 24), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_out_ptr0 + (16 * x0 + 25), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_out_ptr0 + (16 * x0 + 16), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_out_ptr0 + (16 * x0 + 17), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_out_ptr0 + (16 * x0 + 32), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_out_ptr0 + (16 * x0 + 33), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_out_ptr0 + (16 * x0 + 48), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_out_ptr0 + (16 * x0 + 49), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_out_ptr0 + (16 * x0 + 64), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_out_ptr0 + (16 * x0 + 65), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_out_ptr0 + (16 * x0 + 80), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_out_ptr0 + (16 * x0 + 81), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_out_ptr0 + (16 * x0 + 96), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_out_ptr0 + (16 * x0 + 97), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_out_ptr0 + (16 * x0 + 112), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_out_ptr0 + (16 * x0 + 113), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_out_ptr0 + (16 * x0 + 128), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_out_ptr0 + (16 * x0 + 129), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_out_ptr0 + (16 * x0 + 144), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_out_ptr0 + (16 * x0 + 145), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_out_ptr0 + (16 * x0 + 160), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_out_ptr0 + (16 * x0 + 161), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_out_ptr0 + (16 * x0 + 176), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_out_ptr0 + (16 * x0 + 177), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_out_ptr0 + (16 * x0 + 192), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_out_ptr0 + (16 * x0 + 193), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_out_ptr0 + (16 * x0 + 208), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_out_ptr0 + (16 * x0 + 209), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_out_ptr0 + (16 * x0 + 224), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_out_ptr0 + (16 * x0 + 225), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_out_ptr0 + (16 * x0 + 240), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_out_ptr0 + (16 * x0 + 241), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_out_ptr0 + (16 * x0 + 256), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_out_ptr0 + (16 * x0 + 257), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_out_ptr0 + (16 * x0 + 272), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_out_ptr0 + (16 * x0 + 273), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_out_ptr0 + (16 * x0 + 288), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_out_ptr0 + (16 * x0 + 289), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_out_ptr0 + (16 * x0 + 304), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_out_ptr0 + (16 * x0 + 305), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_out_ptr0 + (16 * x0 + 320), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_out_ptr0 + (16 * x0 + 321), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_out_ptr0 + (16 * x0 + 336), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_out_ptr0 + (16 * x0 + 337), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_out_ptr0 + (16 * x0 + 352), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_out_ptr0 + (16 * x0 + 353), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_out_ptr0 + (16 * x0 + 368), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_out_ptr0 + (16 * x0 + 369), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_out_ptr0 + (16 * x0 + 384), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_out_ptr0 + (16 * x0 + 385), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_out_ptr0 + (16 * x0 + 400), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_out_ptr0 + (16 * x0 + 401), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_out_ptr0 + (16 * x0 + 416), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_out_ptr0 + (16 * x0 + 417), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_out_ptr0 + (16 * x0 + 432), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_out_ptr0 + (16 * x0 + 433), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_out_ptr0 + (16 * x0 + 448), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_out_ptr0 + (16 * x0 + 449), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_out_ptr0 + (16 * x0 + 464), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_out_ptr0 + (16 * x0 + 465), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_out_ptr0 + (16 * x0 + 480), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_out_ptr0 + (16 * x0 + 481), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_out_ptr0 + (16 * x0 + 496), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_out_ptr0 + (16 * x0 + 497), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_out_ptr0 + (16 * x0 + 512), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_out_ptr0 + (16 * x0 + 513), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_out_ptr0 + (16 * x0 + 528), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_out_ptr0 + (16 * x0 + 529), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_out_ptr0 + (16 * x0 + 544), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_out_ptr0 + (16 * x0 + 545), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_out_ptr0 + (16 * x0 + 560), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_out_ptr0 + (16 * x0 + 561), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_out_ptr0 + (16 * x0 + 576), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_out_ptr0 + (16 * x0 + 577), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_out_ptr0 + (16 * x0 + 592), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_out_ptr0 + (16 * x0 + 593), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_out_ptr0 + (16 * x0 + 608), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_out_ptr0 + (16 * x0 + 609), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_out_ptr0 + (16 * x0 + 624), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_out_ptr0 + (16 * x0 + 625), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_out_ptr0 + (16 * x0 + 640), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_out_ptr0 + (16 * x0 + 641), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_out_ptr0 + (16 * x0 + 656), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_out_ptr0 + (16 * x0 + 657), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_out_ptr0 + (16 * x0 + 672), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_out_ptr0 + (16 * x0 + 673), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_out_ptr0 + (16 * x0 + 688), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_out_ptr0 + (16 * x0 + 689), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_out_ptr0 + (16 * x0 + 704), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_out_ptr0 + (16 * x0 + 705), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_out_ptr0 + (16 * x0 + 720), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_out_ptr0 + (16 * x0 + 721), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_out_ptr0 + (16 * x0 + 736), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_out_ptr0 + (16 * x0 + 737), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_out_ptr0 + (16 * x0 + 752), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_out_ptr0 + (16 * x0 + 753), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_out_ptr0 + (16 * x0 + 768), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_out_ptr0 + (16 * x0 + 769), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_out_ptr0 + (16 * x0 + 784), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_out_ptr0 + (16 * x0 + 785), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_out_ptr0 + (16 * x0 + 800), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_out_ptr0 + (16 * x0 + 801), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_out_ptr0 + (16 * x0 + 816), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_out_ptr0 + (16 * x0 + 817), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_out_ptr0 + (16 * x0 + 832), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_out_ptr0 + (16 * x0 + 833), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_out_ptr0 + (16 * x0 + 848), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_out_ptr0 + (16 * x0 + 849), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_out_ptr0 + (16 * x0 + 864), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_out_ptr0 + (16 * x0 + 865), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_out_ptr0 + (16 * x0 + 880), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_out_ptr0 + (16 * x0 + 881), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_out_ptr0 + (16 * x0 + 896), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_out_ptr0 + (16 * x0 + 897), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_out_ptr0 + (16 * x0 + 912), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_out_ptr0 + (16 * x0 + 913), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_out_ptr0 + (16 * x0 + 928), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_out_ptr0 + (16 * x0 + 929), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_out_ptr0 + (16 * x0 + 944), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_out_ptr0 + (16 * x0 + 945), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_out_ptr0 + (16 * x0 + 960), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_out_ptr0 + (16 * x0 + 961), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_out_ptr0 + (16 * x0 + 976), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_out_ptr0 + (16 * x0 + 977), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_out_ptr0 + (16 * x0 + 992), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_out_ptr0 + (16 * x0 + 993), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_out_ptr0 + (16 * x0 + 1008), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_out_ptr0 + (16 * x0 + 1009), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_out_ptr0 + (16 * x0 + 1024), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_out_ptr0 + (16 * x0 + 1025), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_out_ptr0 + (16 * x0 + 1040), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_out_ptr0 + (16 * x0 + 1041), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_out_ptr0 + (16 * x0 + 1056), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_out_ptr0 + (16 * x0 + 1057), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_out_ptr0 + (16 * x0 + 1072), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_out_ptr0 + (16 * x0 + 1073), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_out_ptr0 + (16 * x0 + 1088), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_out_ptr0 + (16 * x0 + 1089), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_out_ptr0 + (16 * x0 + 1104), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_out_ptr0 + (16 * x0 + 1105), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_out_ptr0 + (16 * x0 + 1120), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_out_ptr0 + (16 * x0 + 1121), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_out_ptr0 + (16 * x0 + 1136), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_out_ptr0 + (16 * x0 + 1137), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_out_ptr0 + (16 * x0 + 1152), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_out_ptr0 + (16 * x0 + 1153), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_out_ptr0 + (16 * x0 + 1168), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_out_ptr0 + (16 * x0 + 1169), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_out_ptr0 + (16 * x0 + 1184), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_out_ptr0 + (16 * x0 + 1185), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_out_ptr0 + (16 * x0 + 1200), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_out_ptr0 + (16 * x0 + 1201), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_out_ptr0 + (16 * x0 + 1216), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_out_ptr0 + (16 * x0 + 1217), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_out_ptr0 + (16 * x0 + 1232), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_out_ptr0 + (16 * x0 + 1233), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_out_ptr0 + (16 * x0 + 1248), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_out_ptr0 + (16 * x0 + 1249), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_out_ptr0 + (16 * x0 + 1264), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_out_ptr0 + (16 * x0 + 1265), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_out_ptr0 + (16 * x0 + 1280), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_out_ptr0 + (16 * x0 + 1281), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_out_ptr0 + (16 * x0 + 1296), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_out_ptr0 + (16 * x0 + 1297), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_out_ptr0 + (16 * x0 + 1312), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_out_ptr0 + (16 * x0 + 1313), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_out_ptr0 + (16 * x0 + 1328), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_out_ptr0 + (16 * x0 + 1329), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_out_ptr0 + (16 * x0 + 1344), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_out_ptr0 + (16 * x0 + 1345), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_out_ptr0 + (16 * x0 + 1360), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_out_ptr0 + (16 * x0 + 1361), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_out_ptr0 + (16 * x0 + 1376), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_out_ptr0 + (16 * x0 + 1377), xmask, eviction_policy=
        'evict_last')
    tmp274 = tl.load(in_out_ptr0 + (16 * x0 + 1392), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_out_ptr0 + (16 * x0 + 1393), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_out_ptr0 + (16 * x0 + 1408), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_out_ptr0 + (16 * x0 + 1409), xmask,