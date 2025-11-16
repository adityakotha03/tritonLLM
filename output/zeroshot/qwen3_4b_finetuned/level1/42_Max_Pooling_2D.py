import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (1 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (129 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (128 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (129 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (128 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (512 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (513 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (512 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr0 + (513 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr0 + (257 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (258 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr0 + (257 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr0 + (258 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr0 + (127 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr0 + (126 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr0 + (127 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (126 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr0 + (769 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr0 + (770 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp140 = tl.load(in_ptr0 + (769 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr0 + (770 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr0 + (511 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp155 = tl.load(in_ptr0 + (510 + 512 * x0 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (511 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr0 + (510 + 512 * x0 + 1024 * (x2 + 1)), xmask,
        eviction_policy='evict_last')
    tmp0_1 = tl.full([1], 1, tl.int64)
    tmp1_1 = tmp1 > tmp0_1
    tmp2 = tl.where(tmp1_1, tmp1, tmp0)
    tmp3 = tmp0 == tmp2
    tmp4 = tmp1 == tmp2
    tmp5 = tmp0 != tmp2
    tmp6 = tmp4 | tmp5
    tmp7 = tmp3 & tmp6
    tmp8 = tl.where(tmp1_1, tmp0, tmp1)
    tmp9 = tmp1 < tmp0_1
    tmp10 = tl.where(tmp9, tmp1, tmp0)
    tmp11_1 = tmp12 > tmp0_1
    tmp12_1 = tmp11 > tmp10
    tmp13 = tl.where(tmp11_1, tmp12, tmp0)
    tmp14 = tmp0 == tmp13
    tmp15 = tmp12 == tmp13
    tmp16 = tmp0 != tmp13
    tmp17 = tmp15 | tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp13 == tmp8
    tmp20_1 = tmp21 > tmp0_1
    tmp21_1 = tmp20 > tmp10
    tmp22 = tl.where(tmp20_1, tmp21, tmp0)
    tmp23 = tmp0 == tmp22
    tmp24 = tmp21 == tmp22
    tmp25 = tmp0 != tmp22
    tmp26 = tmp24 | tmp25
    tmp27 = tmp23 & tmp26
    tmp28 = tmp22 == tmp8
    tmp29 = tmp21 < tmp0_1
    tmp30 = tl.where(tmp29, tmp21, tmp0)
    tmp31 = tmp21 > tmp10
    tmp32 = tl.where(tmp31, tmp21, tmp10)
    tmp33_1 = tmp34 > tmp0_1
    tmp34_1 = tmp33 > tmp30
    tmp35 = tl.where(tmp33_1, tmp34, tmp0)
    tmp36 = tmp0 == tmp35
    tmp37 = tmp34 == tmp35
    tmp38 = tmp0 != tmp35
    tmp39 = tmp37 | tmp38
    tmp40 = tmp36 & tmp39
    tmp41_1 = tmp42 > tmp0_1
    tmp42_1 = tmp41 > tmp30
    tmp43 = tl.where(tmp41_1, tmp42, tmp0)
    tmp44 = tmp0 == tmp43
    tmp45 = tmp42 == tmp43
    tmp46 = tmp0 != tmp43
    tmp47 = tmp45 | tmp46
    tmp48 = tmp44 & tmp47
    tmp49 = tmp43 == tmp8
    tmp50 = tmp42 < tmp0_1
    tmp51 = tl.where(tmp50, tmp42, tmp0)
    tmp52 = tmp42 > tmp30
    tmp53 = tl.where(tmp52, tmp42, tmp30)
    tmp54 = tmp51 > tmp8
    tmp55_1 = tmp56 > tmp0_1
    tmp56_1 = tmp55 > tmp53
    tmp57 = tl.where(tmp55_1, tmp56, tmp0)
    tmp58 = tmp0 == tmp57
    tmp59 = tmp56 == tmp57
    tmp60 = tmp0 != tmp57
    tmp61 = tmp59 | tmp60
    tmp62 = tmp58 & tmp61
    tmp63 = tmp57 == tmp8
    tmp64 = tmp56 < tmp0_1
    tmp65 = tl.where(tmp64, tmp56, tmp0)
    tmp66 = tmp56 > tmp53
    tmp67 = tl.where(tmp66, tmp56, tmp53)
    tmp68 = tmp65 > tmp8
    tmp69_1 = tmp70 > tmp0_1
    tmp70_1 = tmp68 > tmp67
    tmp71 = tl.where(tmp69_1, tmp70, tmp0)
    tmp72 = tmp0 == tmp71
    tmp73 = tmp70 == tmp71
    tmp74 = tmp0 != tmp71
    tmp75 = tmp73 | tmp74
    tmp76 = tmp72 & tmp75
    tmp77 = tmp71 == tmp8
    tmp78 = tmp70 < tmp0_1
    tmp79 = tl.where(tmp78, tmp70, tmp0)
    tmp80 = tmp70 > tmp67
    tmp81 = tl.where(tmp80, tmp70, tmp67)
    tmp82 = tmp79 > tmp8
    tmp83_1 = tmp84 > tmp0_1
    tmp84_1 = tmp82 > tmp81
    tmp85 = tl.where(tmp83_1, tmp84, tmp0)
    tmp86 = tmp0 == tmp85
    tmp87 = tmp84 == tmp85
    tmp88 = tmp0 != tmp85
    tmp89 = tmp87 | tmp88
    tmp90 = tmp86 & tmp89
    tmp91 = tmp85 == tmp8
    tmp92 = tmp84 < tmp0_1
    tmp93 = tl.where(tmp92, tmp84, tmp0)
    tmp94 = tmp84 > tmp81
    tmp95 = tl.where(tmp94, tmp84, tmp81)
    tmp96 = tmp93 > tmp8
    tmp97 = tmp95 > tmp8
    tmp98_1 = tmp99 > tmp0_1
    tmp99_1 = tmp97 > tmp96
    tmp100 = tl.where(tmp98_1, tmp99, tmp0)
    tmp101 = tmp0 == tmp100
    tmp102 = tmp99 == tmp100
    tmp103 = tmp0 != tmp100
    tmp104 = tmp102 | tmp103
    tmp105 = tmp101 & tmp104
    tmp106 = tmp100 == tmp8
    tmp107 = tmp99 < tmp0_1
    tmp108 = tl.where(tmp107, tmp99, tmp0)
    tmp109 = tmp99 > tmp96
    tmp110 = tl.where(tmp109, tmp99, tmp96)
    tmp111 = tmp108 > tmp8
    tmp112_1 = tmp113 > tmp0_1
    tmp113_1 = tmp111 > tmp110
    tmp114 = tl.where(tmp112_1, tmp113, tmp0)
    tmp115 = tmp0 == tmp114
    tmp116 = tmp113 == tmp114
    tmp117 = tmp0 != tmp114
    tmp118 = tmp116 | tmp117
    tmp119 = tmp115 & tmp118
    tmp120 = tmp114 == tmp8
    tmp121 = tmp113 < tmp0_1
    tmp122 = tl.where(tmp121, tmp113, tmp0)
    tmp123 = tmp113 > tmp110
    tmp124 = tl.where(tmp123, tmp113, tmp110)
    tmp125 = tmp122 > tmp8
    tmp126_1 = tmp127 > tmp0_1
    tmp127_1 = tmp125 > tmp124
    tmp128 = tl.where(tmp126_1, tmp127, tmp0)
    tmp129 = tmp0 == tmp128
    tmp130 = tmp127 == tmp128
    tmp131 = tmp0 != tmp128
    tmp132 = tmp130 | tmp131
    tmp133 = tmp129 & tmp132
    tmp134 = tmp128 == tmp8
    tmp135 = tmp127 < tmp0_1
    tmp136 = tl.where(tmp135, tmp127, tmp0)
    tmp137 = tmp127 > tmp124
    tmp138 = tl.where(tmp137, tmp127, tmp124)
    tmp139 = tmp136 > tmp8
    tmp140_1 = tmp141 > tmp0_1
    tmp141_1 = tmp139 > tmp138
    tmp142 = tl.where(tmp140_1, tmp141, tmp0)
    tmp143 = tmp0 == tmp142
    tmp144 = tmp141 == tmp142
    tmp145 = tmp0 != tmp142
    tmp146 = tmp144 | tmp145
    tmp147 = tmp143 & tmp146
    tmp148 = tmp142 == tmp8
    tmp149 = tmp141 < tmp0_1
    tmp150 = tl.where(tmp149, tmp141, tmp0)
    tmp151 = tmp141 > tmp138
    tmp152 = tl.where(tmp151, tmp141, tmp138)
    tmp153 = tmp150 > tmp8
    tmp154_1 = tmp155 > tmp0_1
    tmp155_1 = tmp153 > tmp152
    tmp156 = tl.where(tmp154_1, tmp155, tmp0)
    tmp157 = tmp0 == tmp156
    tmp158 = tmp155 == tmp156
    tmp159 = tmp0 != tmp156
    tmp160 = tmp158 | tmp159
    tmp161 = tmp157 & tmp160
    tmp162 = tmp156 == tmp8
    tmp163 = tmp155 < tmp0_1
    tmp164 = tl.where(tmp163, tmp155, tmp0)
    tmp165 = tmp155 > tmp152
    tmp166 = tl.where(tmp165, tmp155, tmp152)
    tmp167 = tmp164 > tmp8
    tmp168_1 = tmp169 > tmp0_1
    tmp169_1 = tmp167 > tmp166
    tmp170 = tl.where(tmp168_1, tmp169, tmp0)
    tmp171 = tmp0 == tmp170
    tmp172 = tmp169 == tmp170
    tmp173 = tmp0 != tmp170
    tmp174 = tmp172 | tmp173
    tmp175 = tmp171 & tmp174
    tmp176 = tmp170 == tmp8
    tmp177 = tmp169 < tmp0_1
    tmp178 = tl.where(tmp177, tmp169, tmp0)
    tmp179 = tmp169 > tmp166
    tmp180 = tl.where(tmp179, tmp169, tmp166)
    tmp181 = tmp178 > tmp8
    tl.store(out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr1 + x3, tmp7, xmask)
    tl.store(out_ptr0 + (512 + x3), tmp28, xmask)
    tl.store(out_ptr1 + (512 + x3), tmp48, xmask)
    tl.store(out_ptr0 + (1024 + x3), tmp77, xmask)
    tl.store(out_ptr1 + (1024 + x3), tmp119, xmask)
    tl.store(out_ptr0 + (1536 + x3), tmp157, xmask)
    tl.store(out_ptr1 + (1536 + x3), tmp175, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32, 64, 512, 512), (16384, 256, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((32, 64, 512, 512), (16384, 256, 256, 1),
            torch.float32)
        buf2 = empty_strided_cuda((32, 64, 512, 512), (16384, 256, 256, 1),
            torch.int64)
        get_raw_stream(0)
        triton_per_fused_max_pool2d_with_indices_0[grid(8388608)](arg0_1,
            buf1, buf2, 8388608, XBLOCK=2048, num_warps=16, num_stages=1)
        del arg0_1
    return buf1, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
