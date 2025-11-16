import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 2048
    x1 = xindex % 2048 // 11
    x0 = xindex % 11
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0
    tmp4 = tmp3 < 11
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + 16777216 * x0), tmp5 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (x2 + 2048 * x1 + 16777216 * x0), tmp5 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7 + tmp6
    tmp9 = tmp3 >= 11
    tmp10 = tmp9 & tmp2
    tmp11 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 11) + 16777216 * x0),
        tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp8 + tmp11
    tmp13 = tmp9 & tmp4
    tmp14 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 11) + 16777216 * x0),
        tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp3 >= 22
    tmp17 = tmp16 & tmp2
    tmp18 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 22) + 16777216 * x0),
        tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp15 + tmp18
    tmp20 = tmp16 & tmp4
    tmp21 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 22) + 16777216 * x0),
        tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp3 >= 33
    tmp24 = tmp23 & tmp2
    tmp25 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 33) + 16777216 * x0),
        tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp22 + tmp25
    tmp27 = tmp23 & tmp4
    tmp28 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 33) + 16777216 * x0),
        tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp3 >= 44
    tmp31 = tmp30 & tmp2
    tmp32 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 44) + 16777216 * x0),
        tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp29 + tmp32
    tmp34 = tmp30 & tmp4
    tmp35 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 44) + 16777216 * x0),
        tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp3 >= 55
    tmp38 = tmp37 & tmp2
    tmp39 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 55) + 16777216 * x0),
        tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp36 + tmp39
    tmp41 = tmp37 & tmp4
    tmp42 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 55) + 16777216 * x0),
        tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp40 + tmp42
    tmp44 = tmp3 >= 66
    tmp45 = tmp44 & tmp2
    tmp46 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 66) + 16777216 * x0),
        tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp43 + tmp46
    tmp48 = tmp44 & tmp4
    tmp49 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 66) + 16777216 * x0),
        tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp47 + tmp49
    tmp51 = tmp3 >= 77
    tmp52 = tmp51 & tmp2
    tmp53 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 77) + 16777216 * x0),
        tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tmp50 + tmp53
    tmp55 = tmp51 & tmp4
    tmp56 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 77) + 16777216 * x0),
        tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp54 + tmp56
    tmp58 = tmp3 >= 88
    tmp59 = tmp58 & tmp2
    tmp60 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 88) + 16777216 * x0),
        tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp57 + tmp60
    tmp62 = tmp58 & tmp4
    tmp63 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 88) + 16777216 * x0),
        tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp61 + tmp63
    tmp65 = tmp3 >= 99
    tmp66 = tmp65 & tmp2
    tmp67 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 99) + 16777216 * x0),
        tmp66 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp64 + tmp67
    tmp69 = tmp65 & tmp4
    tmp70 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 99) + 16777216 * x0),
        tmp69 & xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tmp68 + tmp70
    tmp72 = tmp3 >= 110
    tmp73 = tmp72 & tmp2
    tmp74 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 110) + 16777216 * x0),
        tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp71 + tmp74
    tmp76 = tmp72 & tmp4
    tmp77 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 110) + 16777216 * x0),
        tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp75 + tmp77
    tmp79 = tmp3 >= 121
    tmp80 = tmp79 & tmp2
    tmp81 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 121) + 16777216 * x0),
        tmp80 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tmp78 + tmp81
    tmp83 = tmp79 & tmp4
    tmp84 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 121) + 16777216 * x0),
        tmp83 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp82 + tmp84
    tmp86 = tmp3 >= 132
    tmp87 = tmp86 & tmp2
    tmp88 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 132) + 16777216 * x0),
        tmp87 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp85 + tmp88
    tmp90 = tmp86 & tmp4
    tmp91 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 132) + 16777216 * x0),
        tmp90 & xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tmp89 + tmp91
    tmp93 = tmp3 >= 143
    tmp94 = tmp93 & tmp2
    tmp95 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 143) + 16777216 * x0),
        tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp92 + tmp95
    tmp97 = tmp93 & tmp4
    tmp98 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 143) + 16777216 * x0),
        tmp97 & xmask, eviction_policy='evict_last', other=0.0)
    tmp99 = tmp96 + tmp98
    tmp100 = tmp3 >= 154
    tmp101 = tmp100 & tmp2
    tmp102 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 154) + 16777216 * x0),
        tmp101 & xmask, eviction_policy='evict_last', other=0.0)
    tmp103 = tmp99 + tmp102
    tmp104 = tmp100 & tmp4
    tmp105 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 154) + 16777216 * x0),
        tmp104 & xmask, eviction_policy='evict_last', other=0.0)
    tmp106 = tmp103 + tmp105
    tmp107 = tmp3 >= 165
    tmp108 = tmp107 & tmp2
    tmp109 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 165) + 16777216 * x0),
        tmp108 & xmask, eviction_policy='evict_last', other=0.0)
    tmp110 = tmp106 + tmp109
    tmp111 = tmp107 & tmp4
    tmp112 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 165) + 16777216 * x0),
        tmp111 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tmp110 + tmp112
    tmp114 = tmp3 >= 176
    tmp115 = tmp114 & tmp2
    tmp116 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 176) + 16777216 * x0),
        tmp115 & xmask, eviction_policy='evict_last', other=0.0)
    tmp117 = tmp113 + tmp116
    tmp118 = tmp114 & tmp4
    tmp119 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 176) + 16777216 * x0),
        tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp120 = tmp117 + tmp119
    tmp121 = tmp3 >= 187
    tmp122 = tmp121 & tmp2
    tmp123 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 187) + 16777216 * x0),
        tmp122 & xmask, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp120 + tmp123
    tmp125 = tmp121 & tmp4
    tmp126 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 187) + 16777216 * x0),
        tmp125 & xmask, eviction_policy='evict_last', other=0.0)
    tmp127 = tmp124 + tmp126
    tmp128 = tmp3 >= 198
    tmp129 = tmp128 & tmp2
    tmp130 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 198) + 16777216 * x0),
        tmp129 & xmask, eviction_policy='evict_last', other=0.0)
    tmp131 = tmp127 + tmp130
    tmp132 = tmp128 & tmp4
    tmp133 = tl.load(in_ptr0 + (x2 + 2048 * (x1 + 198) + 16777216 * x0),
        tmp132 & xmask, eviction_policy='evict_last', other=0.0)
    tmp134 = tmp131 + tmp133
    tmp135 = tl_math.where(tmp2, tmp121, tmp134)
    tl.store(out_ptr0 + x3, tmp135, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 64, 2048, 2048), (262144, 4096, 2048, 1
        ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 184, 184), (11858048, 184, 184, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(16777216)](arg0_1, buf0, 
            16777216, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
