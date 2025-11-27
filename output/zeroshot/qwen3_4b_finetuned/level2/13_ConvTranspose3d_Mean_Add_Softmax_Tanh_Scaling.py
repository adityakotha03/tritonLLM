import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 3457600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 34576 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused_mean_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp6, xmask)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + x2, tmp3, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 64 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (4 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (5 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (6 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (7 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (8 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (9 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (10 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (11 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (12 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (13 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (14 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (15 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (16 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (17 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (18 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (19 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (20 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (21 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (22 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (23 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (24 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (25 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (26 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (27 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (28 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (29 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (30 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (31 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp5, tmp6)
    tmp7 = triton_helpers.maximum(tmp6, tmp9)
    tmp8 = triton_helpers.maximum(tmp7, tmp11)
    tmp10 = triton_helpers.maximum(tmp8, tmp14)
    tmp12 = triton_helpers.maximum(tmp10, tmp16)
    tmp13 = triton_helpers.maximum(tmp12, tmp19)
    tmp15 = triton_helpers.maximum(tmp13, tmp21)
    tmp17 = triton_helpers.maximum(tmp15, tmp24)
    tmp18 = triton_helpers.maximum(tmp17, tmp26)
    tmp20 = triton_helpers.maximum(tmp18, tmp29)
    tmp22 = triton_helpers.maximum(tmp20, tmp31)
    tmp23 = triton_helpers.maximum(tmp22, tmp34)
    tmp25 = triton_helpers.maximum(tmp23, tmp36)
    tmp27 = triton_helpers.maximum(tmp25, tmp39)
    tmp28 = triton_helpers.maximum(tmp27, tmp41)
    tmp30 = triton_helpers.maximum(tmp28, tmp44)
    tmp32 = triton_helpers.maximum(tmp30, tmp46)
    tmp33 = triton_helpers.maximum(tmp32, tmp49)
    tmp35 = triton_helpers.maximum(tmp33, tmp51)
    tmp37 = triton_helpers.maximum(tmp35, tmp54)
    tmp38 = triton_helpers.maximum(tmp37, tmp56)
    tmp40 = triton_helpers.maximum(tmp38, tmp59)
    tmp42 = triton_helpers.maximum(tmp40, tmp61)
    tmp43 = triton_helpers.maximum(tmp42, tmp64)
    tmp45 = triton_helpers.maximum(tmp43, tmp66)
    tmp47 = triton_helpers.maximum(tmp45, tmp69)
    tmp48 = triton_helpers.maximum(tmp47, tmp71)
    tmp50 = triton_helpers.maximum(tmp48, tmp74)
    tmp52 = triton_helpers.maximum(tmp50, tmp76)
    tmp53 = tmp0 - tmp52
    tmp54 = tl_math.exp(tmp53)
    tl.store(out_ptr0 + x2, tmp54, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 64 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (4 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (5 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (6 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (7 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (8 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (9 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (10 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (11 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (12 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (13 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (14 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (15 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (16 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (17 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (18 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (19 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (20 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (21 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (22 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (23 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (24 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (25 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (26 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (27 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (28 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (29 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (30 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (31 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp2
    tmp4 = tmp2 + tmp4
    tmp5 = tmp4 + tmp6
    tmp6 = tmp5 + tmp9
    tmp7 = tmp6 + tmp11
    tmp8 = tmp7 + tmp14
    tmp9 = tmp8 + tmp16
    tmp10 = tmp9 + tmp19
    tmp11 = tmp10 + tmp21
    tmp12 = tmp11 + tmp24
    tmp13 = tmp12 + tmp26
    tmp14 = tmp13 + tmp29
    tmp15 = tmp14 + tmp31
    tmp16 = tmp15 + tmp34
    tmp17 = tmp16 + tmp36
    tmp18 = tmp17 + tmp39
    tmp19 = tmp18 + tmp41
    tmp20 = tmp19 + tmp44
    tmp21 = tmp20 + tmp46
    tmp22 = tmp21 + tmp49
    tmp23 = tmp22 + tmp51
    tmp24 = tmp23 + tmp54
    tmp25 = tmp24 + tmp56
    tmp26 = tmp25 + tmp59
    tmp27 = tmp26 + tmp61
    tmp28 = tmp27 + tmp64
    tmp29 = tmp28 + tmp66
    tmp30 = tmp29 + tmp69
    tmp31 = tmp30 + tmp71
    tmp32 = tmp31 + tmp74
    tmp33 = tmp32 + tmp76
    tmp34 = tmp0 / tmp33
    tl.store(out_ptr0 + x2, tmp34, xmask)


@triton.jit
def triton_poi_fused_mul_tanh_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 16, 3, 3, 3), (432, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 16, 32, 128, 128), (8388608, 524288,
        16384, 128, 1))
    assert_size_stride(primals_4, (1, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1, 1], dilation=[1, 1, 1], transposed=True, output_padding=[0, 
            0, 0], groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(3457600)](buf2, primals_2, 
            3457600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((16, 1, 1, 1, 1), (1, 16, 16, 16, 16),
            torch.float32)
        buf4 = buf3
        del buf3
        buf5 = buf4
        del buf4
        buf6 = buf5
        del buf5
        triton_per_fused_mean_1[grid(16)](buf6, buf2, buf2, 16, 32, XBLOCK=
            1, num_warps=2, num_stages=1)
        buf7 = empty_strided_cuda((16, 64, 1, 1, 1), (64, 1, 1, 1, 1),
            torch.float32)
        triton_poi_fused_add_2[grid(1024)](buf6, primals_4, buf7, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf8 = empty_strided_cuda((16, 64, 1, 1, 1), (64, 1, 1, 1, 1),
            torch.float32)
        triton_poi_fused__softmax_3[grid(1024)](buf7, buf8, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf9 = buf7
        del buf7
        triton_poi_fused__softmax_4[grid(1024)](buf8, buf9, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf10 = buf8
        del buf8
        triton_poi_fused_mul_tanh_5[grid(1024)](buf9, buf10, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf9
    return buf10, primals_1, primals_3, primals_4, buf2, buf6, buf10


class ModelNew(nn.Module):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling (across depth)
    3. Addition
    4. Softmax (across channels)
    5. Tanh activation
    6. Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
