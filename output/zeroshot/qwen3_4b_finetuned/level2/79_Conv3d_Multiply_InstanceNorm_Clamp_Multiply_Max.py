import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl
    .constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 64 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2048 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_maximum_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp0)
    tmp5 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16 * x3 + 256 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3 + 16 * y2), tmp2, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 3, 16, 32, 32), (16384, 512, 32, 1,
        1))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 3, 16, 32, 32), (12288, 4096, 256,
            8, 1), torch.float32)
        extern_kernels.convolution(primals_2, primals_1, stride=(1, 1, 1),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_1[grid(32768)](buf1, primals_3, 32768,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((128, 16, 15, 31, 31), (74851, 4678, 307,
            9, 1), torch.float32)
        triton_poi_fused_clone_0[grid(512, 64)](primals_5, buf2, 512, 64,
            XBLOCK=64, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_5
        buf4 = extern_kernels.convolution(buf1, buf2, stride=(1, 1, 1),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        buf5 = empty_strided_cuda((128, 16, 15, 31, 31), (74851, 4678, 307,
            9, 1), torch.float32)
        extern_kernels.convolution(buf4, buf2, stride=(1, 1, 1), padding=(
            0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        del buf2
        buf6 = empty_strided_cuda((128, 16), (16, 1), torch.float32)
        triton_poi_fused_maximum_2[grid(16384)](buf4, buf6, 16384, XBLOCK=
            256, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((128, 16, 15, 31, 31), (74851, 4678, 307,
            9, 1), torch.float32)
        triton_poi_fused_convolution_3[grid(512, 16)](buf5, primals_4, buf7,
            512, 16, XBLOCK=16, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_4
    return buf6, primals_1, primals_2, buf1, buf4, buf5, buf6, buf7


class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape,
        clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.conv.bias
        primals_4 = self.multiplier
        primals_5 = self.instance_norm.weight
        primals_6 = self.instance_norm.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6])
        return output[0]
