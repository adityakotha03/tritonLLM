import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16384
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16 * x2 + 2048 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_hardswish_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 * tmp2
    tmp4 = 3.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp7 <= tmp2
    tmp10 = tmp8 <= tmp2
    tmp11 = tmp9 | tmp10
    tmp12 = tl.full([1], 6, tl.int32)
    tmp13 = tmp2 <= tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.where(tmp14, tmp5, tmp6)
    tl.store(out_ptr0 + x0, tmp15, xmask)


@triton.jit
def triton_poi_fused_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 4
    x2 = xindex // 4096
    x4 = xindex % 4
    x5 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3 + 2097152), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (4 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (8 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (12 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x2 + 2097152 * x4), xmask)
    tmp11 = tl.load(in_ptr0 + (1024 + x2 + 2097152 * x4), xmask)
    tmp13 = tl.load(in_ptr0 + (2048 + x2 + 2097152 * x4), xmask)
    tmp15 = tl.load(in_ptr0 + (3072 + x2 + 2097152 * x4), xmask)
    tmp19 = tl.load(in_ptr0 + (x5 + 16384 * x4), xmask)
    tmp20 = tl.load(in_ptr0 + (16384 + x5 + 16384 * x4), xmask)
    tmp22 = tl.load(in_ptr0 + (32768 + x5 + 16384 * x4), xmask)
    tmp24 = tl.load(in_ptr0 + (49152 + x5 + 16384 * x4), xmask)
    tmp28 = tl.load(in_ptr0 + (x3 + 2097152 * x4), xmask)
    tmp29 = tl.load(in_ptr0 + (1024 + x3 + 2097152 * x4), xmask)
    tmp31 = tl.load(in_ptr0 + (2048 + x3 + 2097152 * x4), xmask)
    tmp33 = tl.load(in_ptr0 + (3072 + x3 + 2097152 * x4), xmask)
    tmp37 = tl.load(in_ptr0 + (x5 + 16384 * x4), xmask)
    tmp38 = tl.load(in_ptr0 + (16384 + x5 + 16384 * x4), xmask)
    tmp40 = tl.load(in_ptr0 + (32768 + x5 + 16384 * x4), xmask)
    tmp42 = tl.load(in_ptr0 + (49152 + x5 + 16384 * x4), xmask)
    tmp46 = tl.load(in_ptr0 + (x3 + 2097152 * x4), xmask)
    tmp47 = tl.load(in_ptr0 + (1024 + x3 + 2097152 * x4), xmask)
    tmp49 = tl.load(in_ptr0 + (2048 + x3 + 2097152 * x4), xmask)
    tmp51 = tl.load(in_ptr0 + (3072 + x3 + 2097152 * x4), xmask)
    tmp55 = tl.load(in_ptr0 + (x5 + 16384 * x4), xmask)
    tmp56 = tl.load(in_ptr0 + (16384 + x5 + 16384 * x4), xmask)
    tmp58 = tl.load(in_ptr0 + (32768 + x5 + 16384 * x4), xmask)
    tmp60 = tl.load(in_ptr0 + (49152 + x5 + 16384 * x4), xmask)
    tmp3 = tmp1 * tmp1
    tmp5 = tmp2 * tmp2
    tmp7 = tmp3 + tmp5
    tmp8 = tmp4 * tmp4
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp6
    tmp11 = tmp9 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp13 + tmp1
    tmp15 = tmp14 * tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = 1.0 / tmp17
    tmp19 = tmp19 * tmp18
    tmp20 = tmp20 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp22 * tmp18
    tmp23 = tmp21 + tmp22
    tmp24 = tmp24 * tmp18
    tmp25 = tmp23 + tmp24
    tmp26 = 1.0 / tmp25
    tmp27 = tmp26 * tmp18
    tmp28 = tmp28 * tmp27
    tmp29 = tmp29 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp31 * tmp27
    tmp32 = tmp30 + tmp31
    tmp33 = tmp33 * tmp27
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34 * tmp27
    tmp36 = tmp35 * tmp27
    tmp37 = tmp37 * tmp27
    tmp38 = tmp38 * tmp27
    tmp39 = tmp37 + tmp38
    tmp40 = tmp40 * tmp27
    tmp41 = tmp39 + tmp40
    tmp42 = tmp42 * tmp27
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43 * tmp27
    tmp45 = tmp44 * tmp27
    tmp46 = tmp46 * tmp45
    tmp47 = tmp47 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tmp49 * tmp45
    tmp50 = tmp48 + tmp49
    tmp51 = tmp51 * tmp45
    tmp52 = tmp50 + tmp51
    tmp53 = tmp52 * tmp45
    tmp54 = tmp53 * tmp45
    tmp55 = tmp55 * tmp54
    tmp56 = tmp56 * tmp54
    tmp57 = tmp55 + tmp56
    tmp58 = tmp58 * tmp54
    tmp59 = tmp57 + tmp58
    tmp60 = tmp60 * tmp54
    tmp61 = tmp59 + tmp60
    tmp62 = tmp61 * tmp54
    tmp63 = tmp62 * tmp54
    tmp64 = tmp63 * tmp54
    tmp65 = tmp64 * tmp54
    tmp66 = tmp63 + tmp65
    tmp67 = tmp62 + tmp66
    tmp68 = tmp61 + tmp67
    tmp69 = tmp60 + tmp68
    tmp70 = tmp59 + tmp69
    tmp71 = tmp58 + tmp70
    tmp72 = tmp57 + tmp71
    tmp73 = tmp56 + tmp72
    tmp74 = tmp55 + tmp73
    tmp75 = tmp54 + tmp74
    tmp76 = tmp53 + tmp75
    tmp77 = tmp52 + tmp76
    tmp78 = tmp51 + tmp77
    tmp79 = tmp50 + tmp78
    tmp80 = tmp49 + tmp79
    tmp81 = tmp48 + tmp80
    tmp82 = tmp47 + tmp81
    tmp83 = tmp46 + tmp82
    tmp84 = tmp45 + tmp83
    tmp85 = tmp44 + tmp84
    tmp86 = tmp43 + tmp85
    tmp87 = tmp42 + tmp86
    tmp88 = tmp41 + tmp87
    tmp89 = tmp40 + tmp88
    tmp90 = tmp39 + tmp89
    tmp91 = tmp38 + tmp90
    tmp92 = tmp37 + tmp91
    tmp93 = tmp36 + tmp92
    tmp94 = tmp35 + tmp93
    tmp95 = tmp34 + tmp94
    tmp96 = tmp33 + tmp95
    tmp97 = tmp32 + tmp96
    tmp98 = tmp31 + tmp97
    tmp99 = tmp30 + tmp98
    tmp100 = tmp29 + tmp99
    tmp101 = tmp28 + tmp100
    tmp102 = tmp27 + tmp101
    tmp103 = tmp26 + tmp102
    tmp104 = tmp25 + tmp103
    tmp105 = tmp24 + tmp104
    tmp106 = tmp23 + tmp105
    tmp107 = tmp22 + tmp106
    tmp108 = tmp21 + tmp107
    tmp109 = tmp20 + tmp108
    tmp110 = tmp19 + tmp109
    tmp111 = tmp18 + tmp110
    tmp112 = tmp17 + tmp111
    tmp113 = tmp16 + tmp112
    tmp114 = tmp15 + tmp113
    tmp115 = tmp14 + tmp114
    tmp116 = tmp13 + tmp115
    tmp117 = tmp12 + tmp116
    tmp118 = tmp117 * tmp27
    tmp119 = tmp0 * tmp118
    tl.store(out_ptr0 + (x3 + 2097152 * x4), tmp119, xmask)
    tl.store(out_ptr1 + (x3 + 2097152 * x4), tmp117, xmask)


@triton.jit
def triton_poi_fused_hardswish_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 * tmp2
    tmp4 = 3.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp7 <= tmp2
    tmp10 = tmp8 <= tmp2
    tmp11 = tmp9 | tmp10
    tmp12 = tl.full([1], 6, tl.int32)
    tmp13 = tmp2 <= tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.where(tmp14, tmp5, tmp6)
    tl.store(out_ptr0 + x0, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1))
    assert_size_stride(arg1_1, (16, 3, 3, 3, 3), (162, 54, 18, 6, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 16, 32, 32), (1048576, 65536, 4096,
            128, 1), torch.float32)
        get_ptr0 = buf0
        triton_poi_fused_convolution_0[ext_out_ptr0](arg0_1, get_ptr0, 16384,
            128, XBLOCK=128, YBLOCK=128, num_warps=8, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((128, 16, 16, 32, 32), (1048576, 65536, 4096,
            128, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 16, 16, 32, 32), (1048576, 65536, 4096,
            128, 1), torch.float32)
        triton_poi_fused_group_norm_2[ext_out_ptr1](buf0, arg1_1, buf1, buf2,
            2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg1_1
        buf3 = empty_strided_cuda((128, 16, 16, 32, 32), (1048576, 65536, 4096,
            128, 1), torch.float32)
        triton_poi_fused_hardswish_1[ext_out_ptr0](buf1, buf3, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 16, 16, 32, 32), (1048576, 65536, 4096,
            128, 1), torch.float32)
        triton_poi_fused_hardswish_3[ext_out_ptr0](buf3, buf4, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf3
    return buf4, buf2, buf1, buf0, get_ptr0


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, input_0):
        arg1_1 = self.conv_transpose.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
