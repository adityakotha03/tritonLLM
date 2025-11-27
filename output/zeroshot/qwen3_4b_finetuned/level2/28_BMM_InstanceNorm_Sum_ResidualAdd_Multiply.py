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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_native_batch_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (8192 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (8193 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (8194 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (8195 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (8196 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (8197 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (8198 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (8199 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16384 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (16385 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (16386 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (16387 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (16388 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (16389 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (16390 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (16391 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24576 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (24577 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (24578 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (24579 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (24580 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (24581 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (24582 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (24583 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 8.0
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp14 + tmp30
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp44 = tmp42 + tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp31 + tmp46
    tmp50 = tmp48 + tmp49
    tmp52 = tmp50 + tmp51
    tmp54 = tmp52 + tmp53
    tmp56 = tmp54 + tmp55
    tmp58 = tmp56 + tmp57
    tmp60 = tmp58 + tmp59
    tmp62 = tmp60 + tmp61
    tmp63 = tmp47 + tmp62
    tmp64 = 8192.0
    tmp65 = tmp63 / tmp64
    tmp66 = tmp0 - tmp65
    tmp67 = tmp66 * tmp66
    tmp68 = tmp1 - tmp65
    tmp69 = tmp68 * tmp68
    tmp70 = tmp67 + tmp69
    tmp71 = tmp3 - tmp65
    tmp72 = tmp71 * tmp71
    tmp73 = tmp70 + tmp72
    tmp74 = tmp5 - tmp65
    tmp75 = tmp74 * tmp74
    tmp76 = tmp73 + tmp75
    tmp77 = tmp7 - tmp65
    tmp78 = tmp77 * tmp77
    tmp79 = tmp76 + tmp78
    tmp80 = tmp9 - tmp65
    tmp81 = tmp80 * tmp80
    tmp82 = tmp79 + tmp81
    tmp83 = tmp11 - tmp65
    tmp84 = tmp83 * tmp83
    tmp85 = tmp82 + tmp84
    tmp86 = tmp13 - tmp65
    tmp87 = tmp86 * tmp86
    tmp88 = tmp85 + tmp87
    tmp89 = tmp16 - tmp65
    tmp90 = tmp89 * tmp89
    tmp91 = tmp17 - tmp65
    tmp92 = tmp91 * tmp91
    tmp93 = tmp90 + tmp92
    tmp94 = tmp19 - tmp65
    tmp95 = tmp94 * tmp94
    tmp96 = tmp93 + tmp95
    tmp97 = tmp21 - tmp65
    tmp98 = tmp97 * tmp97
    tmp99 = tmp96 + tmp98
    tmp100 = tmp23 - tmp65
    tmp101 = tmp100 * tmp100
    tmp102 = tmp99 + tmp101
    tmp103 = tmp25 - tmp65
    tmp104 = tmp103 * tmp103
    tmp105 = tmp102 + tmp104
    tmp106 = tmp27 - tmp65
    tmp107 = tmp106 * tmp106
    tmp108 = tmp105 + tmp107
    tmp109 = tmp29 - tmp65
    tmp110 = tmp109 * tmp109
    tmp111 = tmp108 + tmp110
    tmp112 = tmp32 - tmp65
    tmp113 = tmp112 * tmp112
    tmp114 = tmp33 - tmp65
    tmp115 = tmp114 * tmp114
    tmp116 = tmp113 + tmp115
    tmp117 = tmp35 - tmp65
    tmp118 = tmp117 * tmp117
    tmp119 = tmp116 + tmp118
    tmp120 = tmp37 - tmp65
    tmp121 = tmp120 * tmp120
    tmp122 = tmp119 + tmp121
    tmp123 = tmp39 - tmp65
    tmp124 = tmp123 * tmp123
    tmp125 = tmp122 + tmp124
    tmp126 = tmp41 - tmp65
    tmp127 = tmp126 * tmp126
    tmp128 = tmp125 + tmp127
    tmp129 = tmp43 - tmp65
    tmp130 = tmp129 * tmp129
    tmp131 = tmp128 + tmp130
    tmp132 = tmp45 - tmp65
    tmp133 = tmp132 * tmp132
    tmp134 = tmp131 + tmp133
    tmp135 = tmp48 - tmp65
    tmp136 = tmp135 * tmp135
    tmp137 = tmp49 - tmp65
    tmp138 = tmp137 * tmp137
    tmp139 = tmp136 + tmp138
    tmp140 = tmp51 - tmp65
    tmp141 = tmp140 * tmp140
    tmp142 = tmp139 + tmp141
    tmp143 = tmp53 - tmp65
    tmp144 = tmp143 * tmp143
    tmp145 = tmp142 + tmp144
    tmp146 = tmp55 - tmp65
    tmp147 = tmp146 * tmp146
    tmp148 = tmp145 + tmp147
    tmp149 = tmp57 - tmp65
    tmp150 = tmp149 * tmp149
    tmp151 = tmp148 + tmp150
    tmp152 = tmp59 - tmp65
    tmp153 = tmp152 * tmp152
    tmp154 = tmp151 + tmp153
    tmp155 = tmp61 - tmp65
    tmp156 = tmp155 * tmp155
    tmp157 = tmp154 + tmp156
    tmp158 = tmp88 + tmp111
    tmp159 = tmp158 + tmp134
    tmp160 = tmp159 + tmp157
    tmp161 = tmp160 / tmp64
    tmp162 = tmp87 + tmp110
    tmp163 = tmp162 + tmp133
    tmp164 = tmp163 + tmp156
    tmp165 = tmp164 / tmp64
    tmp166 = tmp86 + tmp109
    tmp167 = tmp166 + tmp132
    tmp168 = tmp167 + tmp155
    tmp169 = tmp168 / tmp64
    tmp170 = tmp85 + tmp108
    tmp171 = tmp170 + tmp131
    tmp172 = tmp171 + tmp154
    tmp173 = tmp172 / tmp64
    tmp174 = tmp84 + tmp107
    tmp175 = tmp174 + tmp130
    tmp176 = tmp175 + tmp153
    tmp177 = tmp176 / tmp64
    tmp178 = tmp83 + tmp106
    tmp179 = tmp178 + tmp129
    tmp180 = tmp179 + tmp152
    tmp181 = tmp180 / tmp64
    tmp182 = tmp82 + tmp105
    tmp183 = tmp182 + tmp128
    tmp184 = tmp183 + tmp151
    tmp185 = tmp184 / tmp64
    tmp186 = tmp81 + tmp104
    tmp187 = tmp186 + tmp127
    tmp188 = tmp187 + tmp149
    tmp189 = tmp188 / tmp64
    tmp190 = tmp80 + tmp103
    tmp191 = tmp190 + tmp126
    tmp192 = tmp191 + tmp147
    tmp193 = tmp192 / tmp64
    tmp194 = tmp79 + tmp102
    tmp195 = tmp194 + tmp125
    tmp196 = tmp195 + tmp148
    tmp197 = tmp196 / tmp64
    tmp198 = tmp78 + tmp101
    tmp199 = tmp198 + tmp124
    tmp200 = tmp199 + tmp150
    tmp201 = tmp200 / tmp64
    tmp202 = tmp77 + tmp100
    tmp203 = tmp202 + tmp123
    tmp204 = tmp203 + tmp146
    tmp205 = tmp204 / tmp64
    tmp206 = tmp76 + tmp99
    tmp207 = tmp206 + tmp122
    tmp208 = tmp207 + tmp145
    tmp209 = tmp208 / tmp64
    tmp210 = tmp75 + tmp98
    tmp211 = tmp210 + tmp121
    tmp212 = tmp211 + tmp144
    tmp213 = tmp212 / tmp64
    tmp214 = tmp74 + tmp97
    tmp215 = tmp214 + tmp120
    tmp216 = tmp215 + tmp143
    tmp217 = tmp216 / tmp64
    tmp218 = tmp73 + tmp96
    tmp219 = tmp218 + tmp119
    tmp220 = tmp219 + tmp142
    tmp221 = tmp220 / tmp64
    tmp222 = tmp72 + tmp95
    tmp223 = tmp222 + tmp118
    tmp224 = tmp223 + tmp141
    tmp225 = tmp224 / tmp64
    tmp226 = tmp71 + tmp94
    tmp227 = tmp226 + tmp117
    tmp228 = tmp227 + tmp140
    tmp229 = tmp228 / tmp64
    tmp230 = tmp70 + tmp93
    tmp231 = tmp230 + tmp116
    tmp232 = tmp231 + tmp139
    tmp233 = tmp232 / tmp64
    tmp234 = tmp69 + tmp92
    tmp235 = tmp234 + tmp115
    tmp236 = tmp235 + tmp138
    tmp237 = tmp236 / tmp64
    tmp238 = tmp68 + tmp91
    tmp239 = tmp238 + tmp114
    tmp240 = tmp239 + tmp137
    tmp241 = tmp240 / tmp64
    tmp242 = tmp67 + tmp90
    tmp243 = tmp242 + tmp113
    tmp244 = tmp243 + tmp140
    tmp245 = tmp244 / tmp64
    tmp246 = tmp66 + tmp89
    tmp247 = tmp246 + tmp112
    tmp248 = tmp247 + tmp136
    tmp249 = tmp248 / tmp64
    tl.store(out_ptr0 + x0, tmp65, xmask)
    tl.store(out_ptr1 + x0, tmp233, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + x0, xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + x0, xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr10 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp15 - tmp14
    tmp18 = 8192.0
    tmp19 = tmp18 - tmp17
    tmp20 = tmp16 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp17 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp14 * tmp24
    tmp26 = tmp25 + tmp16
    tl.store(out_ptr0 + x0, tmp26, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp1
    tl.store(out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (1024, 8192), (8192, 1))
    assert_size_stride(primals_5, (1024,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((1024, 1, 1, 8192), (8192, 8192, 8192, 1),
            torch.float32)
        buf2 = empty_strided_cuda((1024, 1, 1, 8192), (8192, 8192, 8192, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_native_batch_norm_0[grid(8192)](buf0, buf1, buf2, 
            8192, XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((1024, 1, 1, 8192), (8192, 8192, 8192, 1),
            torch.float32)
        triton_poi_fused_native_batch_norm_1[grid(8192)](buf0, buf1, buf2,
            primals_2, primals_5, primals_4, buf1, buf2, buf3, primals_3,
            primals_2, buf3, 8192, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        del buf2
        del primals_2
        del primals_5
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_mul_2[grid(8192)](buf3, primals_4, buf4, 8192,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        del primals_4
    return buf4, primals_3, buf0, buf4


class ModelNew(nn.Module):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0, input_1):
        primals_1 = self.bmm.weight
        primals_2 = self.bmm.bias
        primals_3 = input_0
        primals_4 = input_1
        primals_5 = self.instance_norm.weight
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
