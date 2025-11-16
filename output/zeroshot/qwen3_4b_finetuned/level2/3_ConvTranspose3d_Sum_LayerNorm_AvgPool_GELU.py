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
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 1024 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (384 + x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (640 + x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (768 + x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (896 + x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (1152 + x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (1280 + x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (1408 + x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1536 + x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (1664 + x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (1792 + x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (1920 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tmp0 + tmp3
    tmp13 = tmp6 + tmp9
    tmp14 = tmp11 + tmp13
    tmp16 = tmp12 + tmp15
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18 + tmp21
    tmp20 = tmp17 + tmp19
    tmp22 = tmp24 + tmp27
    tmp23 = tmp20 + tmp22
    tmp25 = tmp30 + tmp33
    tmp26 = tmp23 + tmp25
    tmp28 = tmp36 + tmp39
    tmp29 = tmp26 + tmp28
    tmp31 = tmp42 + tmp45
    tmp32 = tmp29 + tmp31
    tmp34 = 64.0
    tmp35 = tmp32 / tmp34
    tmp37 = tmp0 - tmp35
    tmp38 = tmp37 * tmp37
    tmp40 = tmp12 - tmp35
    tmp41 = tmp40 * tmp40
    tmp43 = tmp18 - tmp35
    tmp44 = tmp43 * tmp43
    tmp46 = tmp24 - tmp35
    tmp47 = tmp46 * tmp46
    tmp48 = tmp30 - tmp35
    tmp49 = tmp48 * tmp48
    tmp50 = tmp36 - tmp35
    tmp51 = tmp50 * tmp50
    tmp52 = tmp42 - tmp35
    tmp53 = tmp52 * tmp52
    tmp54 = tmp45 - tmp35
    tmp55 = tmp54 * tmp54
    tmp56 = tmp38 + tmp41
    tmp57 = tmp56 + tmp44
    tmp58 = tmp57 + tmp47
    tmp59 = tmp58 + tmp49
    tmp60 = tmp59 + tmp51
    tmp61 = tmp60 + tmp53
    tmp62 = tmp61 + tmp55
    tmp63 = 64.0
    tmp64 = tmp62 / tmp63
    tmp65 = tmp63 * tmp64
    tmp66 = 0.0015625
    tmp67 = tmp65 * tmp66
    tmp68 = tmp34 - tmp67
    tl.store(out_ptr0 + x2, tmp35, xmask)
    tl.store(out_ptr1 + x2, tmp67, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (64 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (16 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (8 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (4 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (1 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (0 + x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (2 + x0 + 1024 * (x1 + 1)), xmask,
        eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (1 + x0 + 1024 * (x1 + 1)), xmask,
        eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (0 + x0 + 1024 * (x1 + 1)), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp21 = tmp24 + tmp23
    tmp23 = tmp22 + tmp21
    tmp25 = tmp20 + tmp23
    tmp26 = tmp18 + tmp25
    tmp27 = 1024.0
    tmp28 = tmp26 / tmp27
    tl.store(out_ptr0 + x2, tmp28, xmask)


@triton.jit
def triton_poi_fused_gelu_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.0
    tmp4 = tmp0 * tmp3
    tmp5 = tmp0 * tmp0
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 + tmp3
    tmp9 = triton_helpers.tanh(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 + tmp11
    tmp13 = tmp2 & tmp12
    tl.store(out_ptr0 + x0, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 16, 32, 32), (524288, 16384, 1024,
        32, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (32, 64, 18, 34, 34), (435504, 1024, 2464,
            76, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1048576)](buf1, primals_3, 1048576,
            XBLOCK=512, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((32, 64, 18, 34, 34), (435504, 1, 24233, 
            727, 1), torch.float32)
        buf3 = empty_strided_cuda((32, 64, 18, 34, 34), (435504, 1, 24233, 
            727, 1), torch.float32)
        triton_poi_fused_add_1[grid(1048576)](buf2, primals_4, 1048576,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((64, 18, 34, 34), (1024, 1, 32, 1), torch.
            float32)
        buf5 = empty_strided_cuda((64, 18, 34, 34), (1024, 1, 32, 1), torch.
            float32)
        triton_poi_fused_native_layer_norm_2[grid(2048)](buf2, buf4, buf5, 
            2048, XBLOCK=256, num_warps=4, num_stages=1)
        buf6 = extern_kernels.convolution(buf1, primals_5, stride=(2, 2, 2),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (32, 64, 16, 32, 32), (32768, 1, 2048, 64, 
            1))
        buf7 = empty_strided_cuda((32, 64, 16, 32, 32), (32768, 1, 2048, 
            64, 1), torch.float32)
        triton_poi_fused_avg_pool2d_3[grid(1024)](buf6, buf7, 1024, XBLOCK=
            256, num_warps=4, num_stages=1)
        buf8 = empty_strided_cuda((32, 64, 16, 32, 32), (32768, 1, 2048, 
            64, 1), torch.float32)
        triton_poi_fused_gelu_4[grid(1024)](buf7, buf8, 1024, XBLOCK=256,
            num_warps=4, num_stages=1)
    return buf8, primals_1, primals_2, buf7, buf6, buf5, buf4, buf1, buf2


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, 
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_5 = self.norm.weight
        primals_4 = self.norm.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
