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
def triton_poi_fused_native_group_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1887436864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x2 = xindex // 4096
    x4 = xindex
    tmp17 = x0 < 8
    tmp21 = x2 < 8
    tmp23 = tl.load(in_ptr0 + (x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp24 = tl.load(in_ptr1 + x2, tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp25 = tl.load(in_ptr2 + x2, tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tl.where(tmp21, tmp26, tmp26)
    tmp28 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp29 = tl.where(tmp21, tmp28, tmp28)
    tmp30 = tl.load(in_ptr0 + (8 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp31 = tl.load(in_ptr1 + (8 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp32 = tl.load(in_ptr2 + (8 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.where(tmp21, tmp33, tmp33)
    tmp35 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp36 = tl.where(tmp21, tmp35, tmp35)
    tmp37 = tl.load(in_ptr0 + (16 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp38 = tl.load(in_ptr1 + (16 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp39 = tl.load(in_ptr2 + (16 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tl.where(tmp21, tmp40, tmp40)
    tmp42 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp43 = tl.where(tmp21, tmp42, tmp42)
    tmp44 = tl.load(in_ptr0 + (24 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp45 = tl.load(in_ptr1 + (24 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp46 = tl.load(in_ptr2 + (24 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp48 = tl.where(tmp21, tmp47, tmp47)
    tmp49 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp50 = tl.where(tmp21, tmp49, tmp49)
    tmp51 = tl.load(in_ptr0 + (32 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp52 = tl.load(in_ptr1 + (32 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp53 = tl.load(in_ptr2 + (32 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
    tmp55 = tl.where(tmp21, tmp54, tmp54)
    tmp56 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp57 = tl.where(tmp21, tmp56, tmp56)
    tmp58 = tl.load(in_ptr0 + (40 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp59 = tl.load(in_ptr1 + (40 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp60 = tl.load(in_ptr2 + (40 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
    tmp62 = tl.where(tmp21, tmp61, tmp61)
    tmp63 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp64 = tl.where(tmp21, tmp63, tmp63)
    tmp65 = tl.load(in_ptr0 + (48 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp66 = tl.load(in_ptr1 + (48 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp67 = tl.load(in_ptr2 + (48 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK])
    tmp69 = tl.where(tmp21, tmp68, tmp68)
    tmp70 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp71 = tl.where(tmp21, tmp70, tmp70)
    tmp72 = tl.load(in_ptr0 + (56 + x0 + 4096 * x2), tmp17 & tmp21, other=0.0)
    tmp73 = tl.load(in_ptr1 + (56 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp74 = tl.load(in_ptr2 + (56 + x2), tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp76 = tl.where(tmp21, tmp75, tmp75)
    tmp77 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp78 = tl.where(tmp21, tmp77, tmp77)
    tmp79 = tl.load(in_ptr0 + x4, xmask, other=0.0)
    tmp80 = tl.load(in_ptr1 + x2, tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp81 = tl.load(in_ptr2 + x2, tmp21, eviction_policy='evict_last',
        other=0.0)
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK])
    tmp83 = tl.where(tmp21, tmp82, tmp82)
    tmp84 = tl.broadcast_to(tmp80, [XBLOCK])
    tmp85 = tl.where(tmp21, tmp84, tmp84)
    tmp86 = tmp23 + tmp27
    tmp87 = tmp28 + tmp29
    tmp88 = tmp30 + tmp31
    tmp89 = tmp32 + tmp33
    tmp90 = tmp34 + tmp35
    tmp91 = tmp36 + tmp37
    tmp92 = tmp38 + tmp39
    tmp93 = tmp40 + tmp41
    tmp94 = tmp42 + tmp43
    tmp95 = tmp44 + tmp45
    tmp96 = tmp46 + tmp47
    tmp97 = tmp48 + tmp49
    tmp98 = tmp50 + tmp51
    tmp99 = tmp52 + tmp53
    tmp100 = tmp54 + tmp55
    tmp101 = tmp56 + tmp57
    tmp102 = tmp58 + tmp59
    tmp103 = tmp60 + tmp61
    tmp104 = tmp62 + tmp63
    tmp105 = tmp64 + tmp65
    tmp106 = tmp66 + tmp67
    tmp107 = tmp68 + tmp69
    tmp108 = tmp70 + tmp71
    tmp109 = tmp72 + tmp73
    tmp110 = tmp74 + tmp75
    tmp111 = tmp76 + tmp77
    tmp112 = tmp78 + tmp79
    tmp113 = tmp80 + tmp81
    tmp114 = tmp82 + tmp83
    tmp115 = tmp84 + tmp85
    tmp116 = tmp112 + tmp113
    tmp117 = tmp114 + tmp115
    tmp118 = tmp116 + tmp117
    tmp119 = 7.0
    tmp120 = tmp118 / tmp119
    tmp121 = tmp120 - tmp85
    tmp122 = tmp121 * tmp121
    tmp123 = tmp122 + tmp120
    tmp124 = tmp123 - tmp115
    tmp125 = tmp124 * tmp121
    tmp126 = tmp125 + tmp120
    tmp127 = tmp126 - tmp115
    tmp128 = tmp127 * tmp121
    tmp129 = tmp128 + tmp120
    tmp130 = tmp129 - tmp115
    tmp131 = tmp130 * tmp121
    tmp132 = tmp131 + tmp120
    tmp133 = tmp132 - tmp115
    tmp134 = tmp133 * tmp121
    tmp135 = tmp134 + tmp120
    tmp136 = tmp135 - tmp115
    tmp137 = libdevice.rsqrt(tmp136)
    tmp138 = tmp79 - tmp85
    tmp139 = tmp138 * tmp137
    tmp140 = tmp139 * tmp83
    tmp141 = tmp140 + tmp82
    tl.store(out_ptr0 + x4, tmp141, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 8), (8, 1))
    assert_size_stride(arg1_1, (112, 64, 512, 512), (18874368, 294912, 512,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 64, 512, 512), (18874368, 294912, 
            512, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_native_group_norm_0[grid(1887436864)](arg0_1,
            arg1_1, arg1_1, buf0, 1887436864, XBLOCK=256, num_warps=4,
            num_stages=1)
        del arg0_1
    return buf0, arg1_1


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, input_0):
        arg0_1 = self.gn.weight
        arg1_1 = self.gn.bias
        arg1_2 = input_0
        output = call([arg0_1, arg1_2])
        return output[0]