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
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = xindex // 40
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 160 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (40 + x0 + 160 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (80 + x0 + 160 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (120 + x0 + 160 * x1), xmask)
    tmp10 = tl.load(in_ptr0 + (160 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (1 + 160 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (41 + 160 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (81 + 160 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (121 + 160 * x1), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (2 * x0 + 320 * x1), xmask)
    tmp25 = tl.load(in_ptr0 + (320 + 2 * x0 + 320 * x1), xmask)
    tmp27 = tl.load(in_ptr0 + (640 + 2 * x0 + 320 * x1), xmask)
    tmp29 = tl.load(in_ptr0 + (960 + 2 * x0 + 320 * x1), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp10 / tmp7
    tmp11 = tmp8 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp17 = tmp16 + tmp15
    tmp19 = tmp18 + tmp17
    tmp20 = 40.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21 + tmp11
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = tmp30 / tmp20
    tmp32 = tmp31 + tmp22
    tmp33 = tmp23 - tmp22
    tmp34 = tmp33 * tmp33
    tmp35 = tmp25 - tmp22
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = tmp27 - tmp22
    tmp39 = tmp38 * tmp38
    tmp40 = tmp37 + tmp39
    tmp41 = tmp29 - tmp22
    tmp42 = tmp41 * tmp41
    tmp43 = tmp40 + tmp42
    tmp44 = 16.0
    tmp45 = tmp43 / tmp44
    tmp46 = libdevice.rsqrt(tmp45)
    tl.store(out_ptr0 + x2, tmp46, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = xindex // 40
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 160 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 160 * x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tl.store(out_ptr0 + x2, tmp3, xmask)
    tl.store(out_ptr1 + x2, tmp1, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 2560
    xnumel = 27
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 9
    x2 = xindex // 9
    y0 = yindex % 40
    y1 = yindex // 40
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 160 * y1 + 6400 * x2 + 19200 * x1), xmask &
        ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 27 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK
    : tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = yindex % 64
    y2 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * x0 + 1024 * y2), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp3 = tl.where(ymask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + (x0 + 16 * y3), tmp6, xmask & ymask)


@triton.jit
def triton_poi_fused_avg_pool2d_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK
    : tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = yindex % 256
    y1 = yindex // 256
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256 * x0 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp3 = tl.where(ymask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 4.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + (x0 + 4 * y2), tmp6, xmask & ymask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 32, 32, 32), (49152, 16384, 512,
        16, 1))
    assert_size_stride(primals_2, (16, 3, 3, 3, 3), (243, 81, 27, 9, 1))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_5, (16,), (1,))
    assert_size_stride(primals_6, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 3, 3, 3, 3), (243, 1, 81, 27, 9), 
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_0[grid(2560)](primals_4,
            buf0, 2560, XBLOCK=256, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((16, 3, 3, 3, 3), (243, 1, 81, 27, 9),
            torch.float32)
        triton_poi_fused__native_batch_norm_legit_1[grid(2560)](primals_2,
            buf0, buf1, buf0, 2560, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
        del primals_2
        buf2 = extern_kernels.convolution(primals_1, buf1, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (64, 16, 33, 34, 34), (3757504, 234793, 70669,
            2065, 64))
        buf3 = buf2
        del buf2
        triton_poi_fused_convolution_2[grid(2560, 27)](primals_3, buf3, 
            2560, 27, XBLOCK=27, YBLOCK=64, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((64, 16, 33, 34, 34), (3757504, 234793,
            70669, 2065, 64), torch.float32)
        extern_kernels.addmm(primals_7, buf3, buf0, alpha=1, beta=1,
            out=buf4)
        del primals_7
        del buf0
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_3[grid(16384)](buf5, primals_5,
            16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
        buf6 = empty_strided_cuda((64, 16, 16, 16, 16), (4194304, 262144,
            16384, 1024, 1), torch.float32)
        triton_poi_fused_avg_pool2d_4[grid(2048, 16)](buf5, buf6, 2048, 16,
            XBLOCK=16, YBLOCK=64, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((64, 16, 4, 4, 4), (4096, 256, 64, 16, 1),
            torch.float32)
        triton_poi_fused_avg_pool2d_5[grid(1024, 4)](buf6, buf7, 1024, 4,
            XBLOCK=4, YBLOCK=128, num_warps=4, num_stages=1)
    return buf7, primals_1, buf5, primals_6, buf1, reinterpret_tensor(buf3,
        (64, 3, 33, 34, 34), (3757504, 234793, 70669, 2065, 64), 0)


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, input_0):
        primals_4 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_2 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_1 = input_0
        primals_6 = self.avg_pool1.bias
        primals_7 = self.avg_pool2.bias
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
