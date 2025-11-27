import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 345600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x2 = xindex % 3840
    x3 = xindex // 3840
    tmp0 = tl.load(in_out_ptr0 + x4, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x4, tmp4, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp40 = tmp39 + tmp38
    tmp42 = tmp41 + tmp40
    tmp44 = tmp43 + tmp42
    tmp46 = tmp45 + tmp44
    tmp48 = tmp47 + tmp46
    tmp50 = tmp49 + tmp48
    tmp52 = tmp51 + tmp50
    tmp54 = tmp53 + tmp52
    tmp56 = tmp55 + tmp54
    tmp58 = tmp57 + tmp56
    tmp60 = tmp59 + tmp58
    tmp62 = tmp61 + tmp60
    tmp63 = 8.0
    tmp64 = tmp62 / tmp63
    tmp65 = tmp1 - tmp64
    tmp66 = tmp65 * tmp65
    tmp67 = tmp0 - tmp64
    tmp68 = tmp67 * tmp67
    tmp69 = tmp66 + tmp68
    tmp70 = tmp3 - tmp64
    tmp71 = tmp70 * tmp70
    tmp72 = tmp69 + tmp71
    tmp73 = tmp5 - tmp64
    tmp74 = tmp73 * tmp73
    tmp75 = tmp72 + tmp74
    tmp76 = tmp7 - tmp64
    tmp77 = tmp76 * tmp76
    tmp78 = tmp75 + tmp77
    tmp79 = tmp9 - tmp64
    tmp80 = tmp79 * tmp79
    tmp81 = tmp78 + tmp80
    tmp82 = tmp11 - tmp64
    tmp83 = tmp82 * tmp82
    tmp84 = tmp81 + tmp83
    tmp85 = tmp13 - tmp64
    tmp86 = tmp85 * tmp85
    tmp87 = tmp84 + tmp86
    tmp88 = tmp15 - tmp64
    tmp89 = tmp88 * tmp88
    tmp90 = tmp87 + tmp89
    tmp91 = tmp17 - tmp64
    tmp92 = tmp91 * tmp91
    tmp93 = tmp90 + tmp92
    tmp94 = tmp19 - tmp64
    tmp95 = tmp94 * tmp94
    tmp96 = tmp93 + tmp95
    tmp97 = tmp21 - tmp64
    tmp98 = tmp97 * tmp97
    tmp99 = tmp96 + tmp98
    tmp100 = tmp23 - tmp64
    tmp101 = tmp100 * tmp100
    tmp102 = tmp99 + tmp101
    tmp103 = tmp25 - tmp64
    tmp104 = tmp103 * tmp103
    tmp105 = tmp102 + tmp104
    tmp106 = tmp27 - tmp64
    tmp107 = tmp106 * tmp106
    tmp108 = tmp105 + tmp107
    tmp109 = tmp29 - tmp64
    tmp110 = tmp109 * tmp109
    tmp111 = tmp108 + tmp110
    tmp112 = tmp31 - tmp64
    tmp113 = tmp112 * tmp112
    tmp114 = tmp111 + tmp113
    tmp115 = tmp33 - tmp64
    tmp116 = tmp115 * tmp115
    tmp117 = tmp114 + tmp116
    tmp118 = tmp35 - tmp64
    tmp119 = tmp118 * tmp118
    tmp120 = tmp117 + tmp119
    tmp121 = tmp37 - tmp64
    tmp122 = tmp121 * tmp121
    tmp123 = tmp120 + tmp122
    tmp124 = tmp39 - tmp64
    tmp125 = tmp124 * tmp124
    tmp126 = tmp123 + tmp125
    tmp127 = tmp41 - tmp64
    tmp128 = tmp127 * tmp127
    tmp129 = tmp126 + tmp128
    tmp130 = tmp43 - tmp64
    tmp131 = tmp130 * tmp130
    tmp132 = tmp129 + tmp131
    tmp133 = tmp45 - tmp64
    tmp134 = tmp133 * tmp133
    tmp135 = tmp132 + tmp134
    tmp136 = tmp47 - tmp64
    tmp137 = tmp136 * tmp136
    tmp138 = tmp135 + tmp137
    tmp139 = tmp49 - tmp64
    tmp140 = tmp139 * tmp139
    tmp141 = tmp138 + tmp140
    tmp142 = tmp51 - tmp64
    tmp143 = tmp142 * tmp142
    tmp144 = tmp141 + tmp143
    tmp145 = tmp53 - tmp64
    tmp146 = tmp145 * tmp145
    tmp147 = tmp144 + tmp146
    tmp148 = tmp55 - tmp64
    tmp149 = tmp148 * tmp148
    tmp150 = tmp147 + tmp149
    tmp151 = tmp57 - tmp64
    tmp152 = tmp151 * tmp151
    tmp153 = tmp150 + tmp152
    tmp154 = tmp59 - tmp64
    tmp155 = tmp154 * tmp154
    tmp156 = tmp153 + tmp155
    tmp157 = tmp61 - tmp64
    tmp158 = tmp157 * tmp157
    tmp159 = tmp156 + tmp158
    tmp160 = 15.0
    tmp161 = tmp159 / tmp160
    tl.store(out_ptr0 + x0, tmp64, xmask)
    tl.store(out_ptr1 + x0, tmp161, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64 % 64
    x0 = xindex % 64
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = tmp8 * tmp6
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + x3, tmp13, xmask)


@triton.jit
def triton_poi_fused_avg_pool3d_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = xindex // 8 % 8
    x2 = xindex // 64 % 8
    x3 = xindex // 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (5 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (6 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (7 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (0 + x0 + 8 * x1 + 64 * x2 + 512 * x3), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.0625
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + x4, tmp16, xmask)


@triton.jit
def triton_poi_fused_gelu_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (32, 32, 16, 32, 32), (524288, 16384, 1024,
        32, 1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64,), (1,))
    assert_size_stride(primals_6, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1,
            32, 16, 32, 32), (524288, 16384, 1024, 32, 1), 0), primals_1, 
            stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1),
            transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (1, 64, 16, 32, 32), (32768, 512, 32, 1, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(345600)](buf1, primals_2, 345600, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((32, 1, 1, 1, 1), (1, 64, 64, 64, 64),
            torch.float32)
        buf3 = empty_strided_cuda((32, 1, 1, 1, 1), (1, 64, 64, 64, 64),
            torch.float32)
        triton_poi_fused_native_layer_norm_1[grid(64)](buf1, buf2, buf3, 64,
            XBLOCK=64, num_warps=1, num_stages=1)
        buf4 = empty_strided_cuda((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), torch
            .float32)
        triton_poi_fused_native_layer_norm_2[grid(2048)](buf1, buf2, buf3,
            primals_4, primals_5, buf4, 2048, XBLOCK=128, num_warps=4,
            num_stages=1)
        del buf2
        del buf3
        del primals_5
        buf5 = empty_strided_cuda((32, 64, 8, 8, 8), (32768, 512, 64, 8, 1),
            torch.float32)
        extern_kernels.avg_pool3d(buf4, [2, 2, 2], [2, 2, 2], [0, 0, 0], 1)
        buf6 = reinterpret_tensor(buf4, (32, 64, 8, 8, 8), (32768, 512, 64, 
            8, 1), 0)
        del buf4
        triton_poi_fused_avg_pool3d_3[grid(32768)](buf5, buf6, 32768, XBLOCK
            =128, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((32, 64, 8, 8, 8), (32768, 512, 64, 8, 1),
            torch.float32)
        triton_poi_fused_gelu_4[grid(32768)](buf6, buf7, 32768, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf6
    return buf7, primals_1, primals_4, primals_6, reinterpret_tensor(primals_3,
        (1, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), 0), buf1, buf5


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.sum_weight
        primals_5 = self.norm.weight
        primals_6 = self.norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6])
        return output[0]
