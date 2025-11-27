import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
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
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30 / 64.0
    tmp32 = tmp0 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tmp1 - tmp31
    tmp35 = tmp34 * tmp34
    tmp36 = tmp33 + tmp35
    tmp37 = tmp3 - tmp31
    tmp38 = tmp37 * tmp37
    tmp39 = tmp36 + tmp38
    tmp40 = tmp5 - tmp31
    tmp41 = tmp40 * tmp40
    tmp42 = tmp39 + tmp41
    tmp43 = tmp7 - tmp31
    tmp44 = tmp43 * tmp43
    tmp45 = tmp42 + tmp44
    tmp46 = tmp9 - tmp31
    tmp47 = tmp46 * tmp46
    tmp48 = tmp45 + tmp47
    tmp49 = tmp11 - tmp31
    tmp50 = tmp49 * tmp49
    tmp51 = tmp48 + tmp50
    tmp52 = tmp13 - tmp31
    tmp53 = tmp52 * tmp52
    tmp54 = tmp51 + tmp53
    tmp55 = tmp15 - tmp31
    tmp56 = tmp55 * tmp55
    tmp57 = tmp54 + tmp56
    tmp58 = tmp17 - tmp31
    tmp59 = tmp58 * tmp58
    tmp60 = tmp57 + tmp59
    tmp61 = tmp19 - tmp31
    tmp62 = tmp61 * tmp61
    tmp63 = tmp60 + tmp62
    tmp64 = tmp21 - tmp31
    tmp65 = tmp64 * tmp64
    tmp66 = tmp63 + tmp65
    tmp67 = tmp23 - tmp31
    tmp68 = tmp67 * tmp67
    tmp69 = tmp66 + tmp68
    tmp70 = tmp25 - tmp31
    tmp71 = tmp70 * tmp70
    tmp72 = tmp69 + tmp71
    tmp73 = tmp27 - tmp31
    tmp74 = tmp73 * tmp73
    tmp75 = tmp72 + tmp74
    tmp76 = tmp29 - tmp31
    tmp77 = tmp76 * tmp76
    tmp78 = tmp75 + tmp77
    tmp79 = 64.0
    tmp80 = tmp78 / tmp79
    tl.store(out_ptr0 + x0, tmp31, xmask)
    tl.store(out_ptr1 + x0, tmp80, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_gelu_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4096
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
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 4, 4, 4), (2048, 64, 16, 4, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (32, 32, 16, 32, 32), (524288, 16384, 512,
        16, 1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (32, 64, 16, 32, 32), (1048576, 16384, 1024,
            32, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(131072)](buf1, primals_2, 
            131072, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((32, 1, 64, 1, 1), (64, 64, 1, 64, 64),
            torch.float32)
        buf3 = empty_strided_cuda((32, 1, 64, 1, 1), (64, 64, 1, 64, 64),
            torch.float32)
        triton_poi_fused_native_layer_norm_1[grid(128)](buf1, buf2, buf3, 
            128, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((32, 64, 16, 32, 32), (1048576, 16384, 
            1024, 32, 1), torch.float32)
        triton_poi_fused_native_layer_norm_2[grid(4096)](buf1, buf2, buf3,
            primals_4, primals_5, buf4, 4096, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf2
        del buf3
        del primals_5
        buf5 = reinterpret_tensor(buf1, (32, 64, 16, 32, 32), (1048576, 16384,
            1024, 32, 1), 0)
        del buf1
        triton_poi_fused_gelu_mul_3[grid(4096)](buf4, buf5, 4096, XBLOCK=
            256, num_warps=4, num_stages=1)
    return buf5, primals_1, primals_3, primals_4, buf4


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.layer_norm.weight
        primals_5 = self.layer_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
