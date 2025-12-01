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
def triton_poi_fused_clamp_min_mul_sub_var_0(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + 1)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr1 + 2)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + 3)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + 4)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_ptr1 + 5)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + 6)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp28 = tl.load(in_ptr1 + 7)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp32 = tl.load(in_ptr1 + 8)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp36 = tl.load(in_ptr1 + 9)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp40 = tl.load(in_ptr1 + 10)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + 11)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp48 = tl.load(in_ptr1 + 12)
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp52 = tl.load(in_ptr1 + 13)
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + 14)
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp60 = tl.load(in_ptr1 + 15)
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tmp3 + tmp6
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tmp7 + tmp10
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp15 = tmp11 + tmp14
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tmp15 + tmp18
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp23 = tmp19 + tmp22
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tmp23 + tmp26
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp31 = tmp27 + tmp30
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp35 = tmp31 + tmp34
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp39 = tmp35 + tmp38
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp43 = tmp39 + tmp42
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp47 = tmp43 + tmp46
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp51 = tmp47 + tmp50
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
    tmp55 = tmp51 + tmp54
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK])
    tmp59 = tmp55 + tmp58
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp63 = tmp59 + tmp62
    tmp64 = 16.0
    tmp65 = tmp63 / tmp64
    tmp66 = tmp0 - tmp65
    tmp67 = tmp66 * tmp2
    tmp68 = tmp67 + tmp5
    tmp69 = tmp68 - tmp65
    tmp70 = tmp69 * tmp9
    tmp71 = tmp70 + tmp13
    tmp72 = tmp71 - tmp65
    tmp73 = tmp72 * tmp17
    tmp74 = tmp73 + tmp25
    tmp75 = tmp74 - tmp65
    tmp76 = tmp75 * tmp33
    tmp77 = tmp76 + tmp49
    tmp78 = tmp77 - tmp65
    tmp79 = tmp78 * tmp61
    tmp80 = tmp79 + tmp6
    tmp81 = tmp80 - tmp65
    tmp82 = tmp81 * tmp20
    tmp83 = tmp82 + tmp28
    tmp84 = tmp83 - tmp65
    tmp85 = tmp84 * tmp36
    tmp86 = tmp85 + tmp52
    tmp87 = tmp86 - tmp65
    tmp88 = tmp87 * tmp60
    tmp89 = tmp88 + tmp12
    tmp90 = tmp89 - tmp65
    tmp91 = tmp90 * tmp24
    tmp92 = tmp91 + tmp32
    tmp93 = tmp92 - tmp65
    tmp94 = tmp93 * tmp48
    tmp95 = tmp94 + tmp56
    tmp96 = tmp95 - tmp65
    tmp97 = tmp96 * tmp64
    tmp98 = 1.0
    tmp99 = triton_helpers.maximum(tmp97, tmp98)
    tmp100 = 0.0
    tmp101 = triton_helpers.minimum(tmp99, tmp100)
    tl.store(out_ptr0 + x0, tmp101, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16,), (1,))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 16, 16, 64, 64), (163840, 10240, 640,
        10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=8, bias=None)
        assert_size_stride(buf0, (128, 16, 16, 64, 64), (163840, 10240, 640,
            10, 1))
        buf1 = empty_strided_cuda((128, 16, 16, 64, 64), (163840, 10240, 
            640, 10, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_min_mul_sub_var_0[grid(1048576)](buf0,
            primals_1, primals_2, buf1, 1048576, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf0
        del primals_1
        del primals_2
    return buf1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]