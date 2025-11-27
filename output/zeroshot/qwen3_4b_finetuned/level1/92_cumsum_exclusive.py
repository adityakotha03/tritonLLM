import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32766
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32766
    x1 = xindex // 32766
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 32766, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32766 * x1 + x0), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32768, tl.int64)
    tmp9 = tl.load(in_ptr0 + (32766 * x1 + (-32766 + x0)), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_cumsum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32766
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (32766 + x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp1 + tmp0
    tmp3 = tl.load(in_ptr0 + (65532 + x0), xmask, eviction_policy='evict_last'
        )
    tmp4 = tmp3 + tmp2
    tmp5 = tl.load(in_ptr0 + (98304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp6 = tmp5 + tmp4
    tmp7 = tl.load(in_ptr0 + (131070 + x0), xmask, eviction_policy=
        'evict_last')
    tmp8 = tmp7 + tmp6
    tmp9 = tl.load(in_ptr0 + (163836 + x0), xmask, eviction_policy=
        'evict_last')
    tmp10 = tmp9 + tmp8
    tmp11 = tl.load(in_ptr0 + (196602 + x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tmp11 + tmp10
    tmp13 = tl.load(in_ptr0 + (229368 + x0), xmask, eviction_policy=
        'evict_last')
    tmp14 = tmp13 + tmp12
    tmp15 = tl.load(in_ptr0 + (262134 + x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tmp15 + tmp14
    tmp17 = tl.load(in_ptr0 + (294900 + x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tmp17 + tmp16
    tmp19 = tl.load(in_ptr0 + (327666 + x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tmp19 + tmp18
    tmp21 = tl.load(in_ptr0 + (360432 + x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tmp21 + tmp20
    tmp23 = tl.load(in_ptr0 + (393198 + x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tmp23 + tmp22
    tmp25 = tl.load(in_ptr0 + (425964 + x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tmp25 + tmp24
    tmp27 = tl.load(in_ptr0 + (458730 + x0), xmask, eviction_policy=
        'evict_last')
    tmp28 = tmp27 + tmp26
    tmp29 = tl.load(in_ptr0 + (491496 + x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tmp29 + tmp28
    tmp31 = tl.load(in_ptr0 + (524262 + x0), xmask, eviction_policy=
        'evict_last')
    tmp32 = tmp31 + tmp30
    tmp33 = tl.load(in_ptr0 + (557028 + x0), xmask, eviction_policy=
        'evict_last')
    tmp34 = tmp33 + tmp32
    tmp35 = tl.load(in_ptr0 + (589794 + x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tmp35 + tmp34
    tmp37 = tl.load(in_ptr0 + (622560 + x0), xmask, eviction_policy=
        'evict_last')
    tmp38 = tmp37 + tmp36
    tmp39 = tl.load(in_ptr0 + (655326 + x0), xmask, eviction_policy=
        'evict_last')
    tmp40 = tmp39 + tmp38
    tmp41 = tl.load(in_ptr0 + (688092 + x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tmp41 + tmp40
    tmp43 = tl.load(in_ptr0 + (720858 + x0), xmask, eviction_policy=
        'evict_last')
    tmp44 = tmp43 + tmp42
    tmp45 = tl.load(in_ptr0 + (753624 + x0), xmask, eviction_policy=
        'evict_last')
    tmp46 = tmp45 + tmp44
    tmp47 = tl.load(in_ptr0 + (786390 + x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tmp47 + tmp46
    tmp49 = tl.load(in_ptr0 + (819156 + x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tmp49 + tmp48
    tmp51 = tl.load(in_ptr0 + (851922 + x0), xmask, eviction_policy=
        'evict_last')
    tmp52 = tmp51 + tmp50
    tmp53 = tl.load(in_ptr0 + (884688 + x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tmp53 + tmp52
    tmp55 = tl.load(in_ptr0 + (917454 + x0), xmask, eviction_policy=
        'evict_last')
    tmp56 = tmp55 + tmp54
    tmp57 = tl.load(in_ptr0 + (950220 + x0), xmask, eviction_policy=
        'evict_last')
    tmp58 = tmp57 + tmp56
    tmp59 = tl.load(in_ptr0 + (982986 + x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tmp59 + tmp58
    tmp61 = tl.load(in_ptr0 + (1015752 + x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tmp61 + tmp60
    tmp63 = tl.load(in_ptr0 + (1048518 + x0), xmask, eviction_policy=
        'evict_last')
    tmp64 = tmp63 + tmp62
    tmp65 = tl.load(in_ptr0 + (1081284 + x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tmp65 + tmp64
    tmp67 = tl.load(in_ptr0 + (1114050 + x0), xmask, eviction_policy=
        'evict_last')
    tmp68 = tmp67 + tmp66
    tmp69 = tl.load(in_ptr0 + (1146816 + x0), xmask, eviction_policy=
        'evict_last')
    tmp70 = tmp69 + tmp68
    tmp71 = tl.load(in_ptr0 + (1179582 + x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tmp71 + tmp70
    tmp73 = tl.load(in_ptr0 + (1212348 + x0), xmask, eviction_policy=
        'evict_last')
    tmp74 = tmp73 + tmp72
    tmp75 = tl.load(in_ptr0 + (1245114 + x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tmp75 + tmp74
    tmp77 = tl.load(in_ptr0 + (1277880 + x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tmp77 + tmp76
    tmp79 = tl.load(in_ptr0 + (1310646 + x0), xmask, eviction_policy=
        'evict_last')
    tmp80 = tmp79 + tmp78
    tmp81 = tl.load(in_ptr0 + (1343412 + x0), xmask, eviction_policy=
        'evict_last')
    tmp82 = tmp81 + tmp80
    tmp83 = tl.load(in_ptr0 + (1376178 + x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tmp83 + tmp82
    tmp85 = tl.load(in_ptr0 + (1408944 + x0), xmask, eviction_policy=
        'evict_last')
    tmp86 = tmp85 + tmp84
    tmp87 = tl.load(in_ptr0 + (1441710 + x0), xmask, eviction_policy=
        'evict_last')
    tmp88 = tmp87 + tmp86
    tmp89 = tl.load(in_ptr0 + (1474476 + x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tmp89 + tmp88
    tmp91 = tl.load(in_ptr0 + (1507242 + x0), xmask, eviction_policy=
        'evict_last')
    tmp92 = tmp91 + tmp90
    tmp93 = tl.load(in_ptr0 + (1540008 + x0), xmask, eviction_policy=
        'evict_last')
    tmp94 = tmp93 + tmp92
    tmp95 = tl.load(in_ptr0 + (1572774 + x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tmp95 + tmp94
    tmp97 = tl.load(in_ptr0 + (1605540 + x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tmp97 + tmp96
    tmp99 = tl.load(in_ptr0 + (1638306 + x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tmp99 + tmp98
    tmp101 = tl.load(in_ptr0 + (1671072 + x0), xmask, eviction_policy=
        'evict_last')
    tmp102 = tmp101 + tmp100
    tmp103 = tl.load(in_ptr0 + (1703838 + x0), xmask, eviction_policy=
        'evict_last')
    tmp104 = tmp103 + tmp102
    tmp105 = tl.load(in_ptr0 + (1736604 + x0), xmask, eviction_policy=
        'evict_last')
    tmp106 = tmp105 + tmp104
    tmp107 = tl.load(in_ptr0 + (1769370 + x0), xmask, eviction_policy=
        'evict_last')
    tmp108 = tmp107 + tmp106
    tmp109 = tl.load(in_ptr0 + (1802136 + x0), xmask, eviction_policy=
        'evict_last')
    tmp110 = tmp109 + tmp108
    tmp111 = tl.load(in_ptr0 + (1834902 + x0), xmask, eviction_policy=
        'evict_last')
    tmp112 = tmp111 + tmp110
    tmp113 = tl.load(in_ptr0 + (1867668 + x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tmp113 + tmp112
    tmp115 = tl.load(in_ptr0 + (1900434 + x0), xmask, eviction_policy=
        'evict_last')
    tmp116 = tmp115 + tmp114
    tmp117 = tl.load(in_ptr0 + (1933200 + x0), xmask, eviction_policy=
        'evict_last')
    tmp118 = tmp117 + tmp116
    tmp119 = tl.load(in_ptr0 + (1965966 + x0), xmask, eviction_policy=
        'evict_last')
    tmp120 = tmp119 + tmp118
    tmp121 = tl.load(in_ptr0 + (1998732 + x0), xmask, eviction_policy=
        'evict_last')
    tmp122 = tmp121 + tmp120
    tmp123 = tl.load(in_ptr0 + (2031498 + x0), xmask, eviction_policy=
        'evict_last')
    tmp124 = tmp123 + tmp122
    tmp125 = tl.load(in_ptr0 + (2064264 + x0), xmask, eviction_policy=
        'evict_last')
    tmp126 = tmp125 + tmp124
    tmp127 = tl.load(in_ptr0 + (2097030 + x0), xmask, eviction_policy=
        'evict_last')
    tmp128 = tmp127 + tmp126
    tmp129 = tl.load(in_ptr0 + (2129796 + x0), xmask, eviction_policy=
        'evict_last')
    tmp130 = tmp129 + tmp128
    tmp131 = tl.load(in_ptr0 + (2162562 + x0), xmask, eviction_policy=
        'evict_last')
    tmp132 = tmp131 + tmp130
    tmp133 = tl.load(in_ptr0 + (2195328 + x0), xmask, eviction_policy=
        'evict_last')
    tmp134 = tmp133 + tmp132
    tmp135 = tl.load(in_ptr0 + (2228094 + x0), xmask, eviction_policy=
        'evict_last')
    tmp136 = tmp135 + tmp134
    tmp137 = tl.load(in_ptr0 + (2260860 + x0), xmask, eviction_policy=
        'evict_last')
    tmp138 = tmp137 + tmp136
    tmp139 = tl.load(in_ptr0 + (2293626 + x0), xmask, eviction_policy=
        'evict_last')
    tmp140 = tmp139 + tmp138
    tmp141 = tl.load(in_ptr0 + (2326392 + x0), xmask, eviction_policy=
        'evict_last')
    tmp142 = tmp141 + tmp140
    tmp143 = tl.load(in_ptr0 + (2359158 + x0), xmask, eviction_policy=
        'evict_last')
    tmp144 = tmp143 + tmp142
    tmp145 = tl.load(in_ptr0 + (2391924 + x0), xmask, eviction_policy=
        'evict_last')
    tmp146 = tmp145 + tmp144
    tmp147 = tl.load(in_ptr0 + (2424690 + x0), xmask, eviction_policy=
        'evict_last')
    tmp148 = tmp147 + tmp146
    tmp149 = tl.load(in_ptr0 + (2457456 + x0), xmask, eviction_policy=
        'evict_last')
    tmp150 = tmp149 + tmp148
    tmp151 = tl.load(in_ptr0 + (2490222 + x0), xmask, eviction_policy=
        'evict_last')
    tmp152 = tmp151 + tmp150
    tmp153 = tl.load(in_ptr0 + (2522988 + x0), xmask, eviction_policy=
        'evict_last')
    tmp154 = tmp153 + tmp152
    tmp155 = tl.load(in_ptr0 + (2555754 + x0), xmask, eviction_policy=
        'evict_last')
    tmp156 = tmp155 + tmp154
    tmp157 = tl.load(in_ptr0 + (2588520 + x0), xmask, eviction_policy=
        'evict_last')
    tmp158 = tmp157 + tmp156
    tmp159 = tl.load(in_ptr0 + (2621286 + x0), xmask, eviction_policy=
        'evict_last')
    tmp160 = tmp159 + tmp158
    tmp161 = tl.load(in_ptr0 + (2654052 + x0), xmask, eviction_policy=
        'evict_last')
    tmp162 = tmp161 + tmp160
    tmp163 = tl.load(in_ptr0 + (2686818 + x0), xmask, eviction_policy=
        'evict_last')
    tmp164 = tmp163 + tmp162
    tmp165 = tl.load(in_ptr0 + (2719584 + x0), xmask, eviction_policy=
        'evict_last')
    tmp166 = tmp165 + tmp164
    tmp167 = tl.load(in_ptr0 + (2752350 + x0), xmask, eviction_policy=
        'evict_last')
    tmp168 = tmp167 + tmp166
    tmp169 = tl.load(in_ptr0 + (2785116 + x0), xmask, eviction_policy=
        'evict_last')
    tmp170 = tmp169 + tmp168
    tmp171 = tl.load(in_ptr0 + (2817882 + x0), xmask, eviction_policy=
        'evict_last')
    tmp172 = tmp171 + tmp170
    tmp173 = tl.load(in_ptr0 + (2850648 + x0), xmask, eviction_policy=
        'evict_last')
    tmp174 = tmp173 + tmp172
    tmp175 = tl.load(in_ptr0 + (2883414 + x0), xmask, eviction_policy=
        'evict_last')
    tmp176 = tmp175 + tmp174
    tmp177 = tl.load(in_ptr0 + (2916180 + x0), xmask, eviction_policy=
        'evict_last')
    tmp178 = tmp177 + tmp176
    tmp179 = tl.load(in_ptr0 + (2948946 + x0), xmask, eviction_policy=
        'evict_last')
    tmp180 = tmp179 + tmp178
    tmp181 = tl.load(in_ptr0 + (2981712 + x0), xmask, eviction_policy=
        'evict_last')
    tmp182 = tmp181 + tmp180
    tmp183 = tl.load(in_ptr0 + (3014478 + x0), xmask, eviction_policy=
        'evict_last')
    tmp184 = tmp183 + tmp182
    tmp185 = tl.load(in_ptr0 + (3047244 + x0), xmask, eviction_policy=
        'evict_last')
    tmp186 = tmp185 + tmp184
    tmp187 = tl.load(in_ptr0 + (3079998 + x0), xmask, eviction_policy=
        'evict_last')
    tmp188 = tmp187 + tmp186
    tmp189 = tl.load(in_ptr0 + (3112764 + x0), xmask, eviction_policy=
        'evict_last')
    tmp190 = tmp189 + tmp188
    tmp191 = tl.load(in_ptr0 + (3145530 + x0), xmask, eviction_policy=
        'evict_last')
    tmp192 = tmp191 + tmp190
    tmp193 = tl.load(in_ptr0 + (3178296 + x0), xmask, eviction_policy=
        'evict_last')
    tmp194 = tmp193 + tmp192
    tmp195 = tl.load(in_ptr0 + (3211062 + x0), xmask, eviction_policy=
        'evict_last')
    tmp196 = tmp195 + tmp194
    tmp197 = tl.load(in_ptr0 + (3243828 + x0), xmask, eviction_policy=
        'evict_last')
    tmp198 = tmp197 + tmp196
    tmp199 = tl.load(in_ptr0 + (3276660 + x0), xmask, eviction_policy=
        'evict_last')
    tmp200 = tmp199 + tmp198
    tl.store(out_ptr0 + x0, tmp200, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32766,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(32766)](arg0_1, buf0, 32766, XBLOCK=
            128, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((32766,), (1,), torch.float32)
        triton_poi_fused_cumsum_1[grid(32766)](buf0, buf1, 32766, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
