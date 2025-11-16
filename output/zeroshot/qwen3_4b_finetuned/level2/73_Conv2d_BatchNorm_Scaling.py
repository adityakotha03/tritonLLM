import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 36 * y0 + 49152 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (y3 + 128 * x2), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, None)


@triton.jit
def triton_poi_fused_batch_norm2d_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x2), xmask)
    tmp7 = tl.load(in_ptr1 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (768 + x2), xmask)
    tmp11 = tl.load(in_ptr1 + (3 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp5 = tmp4 + tmp3
    tmp8 = tmp7 + tmp6
    tmp9 = tmp11 + tmp10
    tmp12 = 16.0
    tmp13 = tmp12 / tmp12
    tmp14 = tmp13 * tmp2
    tmp15 = tmp13 * tmp5
    tmp16 = tmp13 * tmp8
    tmp17 = tmp13 * tmp9
    tmp18 = tmp14 + tmp15
    tmp19 = tmp18 + tmp16
    tmp20 = tmp19 + tmp17
    tmp21 = tmp20 / tmp12
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tmp0 - tmp21
    tmp25 = tmp24 * tmp24
    tmp26 = tmp3 - tmp21
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 + tmp27
    tmp29 = tmp6 - tmp21
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 + tmp30
    tmp32 = tmp10 - tmp21
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 + tmp33
    tmp35 = 4.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp36 * tmp13
    tmp38 = tmp14 - tmp21
    tmp39 = tmp38 * tmp38
    tmp40 = tmp15 - tmp21
    tmp41 = tmp40 * tmp40
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42 / tmp35
    tmp44 = tmp36 * tmp43
    tmp45 = tmp16 - tmp21
    tmp46 = tmp45 * tmp45
    tmp47 = tmp17 - tmp21
    tmp48 = tmp47 * tmp47
    tmp49 = tmp46 + tmp48
    tmp50 = tmp49 / tmp35
    tmp51 = tmp36 * tmp50
    tmp52 = tmp44 + tmp51
    tmp53 = tmp52 / tmp13
    tmp54 = tmp13 * tmp37
    tmp55 = tmp53 - tmp54
    tl.store(out_ptr0 + x2, tmp21, xmask)
    tl.store(out_ptr1 + x2, tmp55, xmask)


@triton.jit
def triton_poi_fused_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 126, 126), (1032192, 16128, 126,
            1))
        buf1 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 64, 1), (64, 1, 128), torch.float32)
        buf3 = empty_strided_cuda((128, 64, 128), (8192, 128, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(128, 36)](primals_1, buf1, 128, 36,
            XBLOCK=16, YBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        triton_poi_fused_convolution_1[grid(49152)](buf0, buf1, 49152,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 64, 126, 126), (1022176, 16128, 126,
            1), torch.float32)
        buf5 = buf0
        del buf0
        extern_kernels.convolution(primals_3, primals_2, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf6 = empty_strided_cuda((128, 64, 126, 126), (1022176, 16128, 126,
            1), torch.float32)
        buf7 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        triton_poi_fused_batch_norm2d_2[grid(32768)](buf5, primals_2, buf6,
            buf7, 32768, XBLOCK=256, num_warps=4, num_stages=1)
        del buf5
        del primals_2
        buf8 = buf3
        del buf3
        triton_poi_fused_mul_3[grid(32768)](buf6, buf7, 32768, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf7
    return buf8, primals_3, buf6, buf2, buf4, buf8


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
