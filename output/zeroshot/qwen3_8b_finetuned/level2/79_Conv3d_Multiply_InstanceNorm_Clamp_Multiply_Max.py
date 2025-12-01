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
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = -1.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 16 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (1 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr1 + 1 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr1 + 2 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (3 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr1 + 3 + x0, xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (4 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr1 + 4 + x0, xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (5 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr1 + 5 + x0, xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (6 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp26 = tl.load(in_ptr1 + 6 + x0, xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (7 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr1 + 7 + x0, xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (8 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr1 + 8 + x0, xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (9 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr1 + 9 + x0, xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (10 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr1 + 10 + x0, xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (11 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr1 + 11 + x0, xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (12 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr1 + 12 + x0, xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (13 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr1 + 13 + x0, xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (14 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr1 + 14 + x0, xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (15 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr1 + 15 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp7 = tmp5 + tmp6
    tmp11 = tmp9 + tmp10
    tmp15 = tmp13 + tmp14
    tmp19 = tmp17 + tmp18
    tmp23 = tmp21 + tmp22
    tmp27 = tmp25 + tmp26
    tmp31 = tmp29 + tmp30
    tmp35 = tmp33 + tmp34
    tmp39 = tmp37 + tmp38
    tmp43 = tmp41 + tmp42
    tmp47 = tmp45 + tmp46
    tmp51 = tmp49 + tmp50
    tmp55 = tmp53 + tmp54
    tmp59 = tmp57 + tmp58
    tmp63 = tmp61 + tmp62
    tmp64 = 16.0
    tmp65 = tmp2 / tmp64
    tmp66 = tmp7 / tmp64
    tmp67 = tmp65 - tmp66
    tmp68 = tmp67 * tmp67
    tmp69 = tmp11 / tmp64
    tmp70 = tmp69 - tmp66
    tmp71 = tmp70 * tmp70
    tmp72 = tmp15 / tmp64
    tmp73 = tmp72 - tmp66
    tmp74 = tmp73 * tmp73
    tmp75 = tmp19 / tmp64
    tmp76 = tmp75 - tmp66
    tmp77 = tmp76 * tmp76
    tmp78 = tmp23 / tmp64
    tmp79 = tmp78 - tmp66
    tmp80 = tmp79 * tmp79
    tmp81 = tmp27 / tmp64
    tmp82 = tmp81 - tmp66
    tmp83 = tmp82 * tmp82
    tmp84 = tmp31 / tmp64
    tmp85 = tmp84 - tmp66
    tmp86 = tmp85 * tmp85
    tmp87 = tmp35 / tmp64
    tmp88 = tmp87 - tmp66
    tmp89 = tmp88 * tmp88
    tmp90 = tmp39 / tmp64
    tmp91 = tmp90 - tmp66
    tmp92 = tmp91 * tmp91
    tmp93 = tmp43 / tmp64
    tmp94 = tmp93 - tmp66
    tmp95 = tmp94 * tmp94
    tmp96 = tmp47 / tmp64
    tmp97 = tmp96 - tmp66
    tmp98 = tmp97 * tmp97
    tmp99 = tmp51 / tmp64
    tmp100 = tmp99 - tmp66
    tmp101 = tmp100 * tmp100
    tmp102 = tmp55 / tmp64
    tmp103 = tmp102 - tmp66
    tmp104 = tmp103 * tmp103
    tmp105 = tmp59 / tmp64
    tmp106 = tmp105 - tmp66
    tmp107 = tmp106 * tmp106
    tmp108 = tmp63 / tmp64
    tmp109 = tmp108 - tmp66
    tmp110 = tmp109 * tmp109
    tmp111 = tmp68 + tmp71
    tmp112 = tmp111 + tmp74
    tmp113 = tmp112 + tmp77
    tmp114 = tmp113 + tmp80
    tmp115 = tmp114 + tmp83
    tmp116 = tmp115 + tmp86
    tmp117 = tmp116 + tmp89
    tmp118 = tmp117 + tmp92
    tmp119 = tmp118 + tmp95
    tmp120 = tmp119 + tmp98
    tmp121 = tmp120 + tmp101
    tmp122 = tmp121 + tmp104
    tmp123 = tmp122 + tmp107
    tmp124 = tmp123 + tmp110
    tmp125 = 15.0
    tmp126 = tmp124 / tmp125
    tmp127 = 1e-05
    tmp128 = tmp126 + tmp127
    tmp129 = libdevice.rsqrt(tmp128)
    tmp130 = tmp64 * tmp129
    tl.store(out_ptr0 + x0, tmp130, xmask)
    tl.store(out_ptr1 + x0, tmp4, xmask)


@triton.jit
def triton_red_fused_max_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 16 * tl.broadcast_to(xoffset, [XBLOCK,
        1])), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(tl.full([XBLOCK, 1], True, tl.int1), tmp1, tmp1)
    tmp4 = triton_helpers.promote_to_tensor(tl.max2(tmp3, 1)[:, None])
    tl.store(out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp4, None)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 16, 32, 32), (16384, 5461, 340, 
        10, 1))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 14, 30, 30), (67200, 4200, 300, 
            10, 1))
        buf1 = empty_strided_cuda((128, 16, 14, 30, 30), (67200, 4200, 300,
            10, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(1881600)](buf0, primals_2, buf1, 1881600,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 16, 1, 1, 1), (16, 1, 16, 16, 16),
            torch.float32)
        buf3 = empty_strided_cuda((128, 16, 1, 1, 1), (16, 1, 16, 16, 16),
            torch.float32)
        triton_poi_fused_1[grid(128)](buf1, primals_1, buf2, buf3, 128,
            XBLOCK=16, num_warps=4, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((128, 1, 14, 30, 30), (12600, 12600, 900,
            30, 1), torch.float32)
        triton_red_fused_max_2[grid(128)](buf3, buf4, 128, 16, XBLOCK=1,
            num_warps=1, num_stages=1)
    return buf4, buf0, buf1, buf3, primals_1


class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        primals_2 = self.multiplier
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]