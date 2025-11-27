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


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 101776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 25444
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256 % 16
    x3 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x2 + 512 * x1 + 8192 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32 * x2 + 512 * x1 + 8192 * x3),
        xmask)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + 32 * x2 + 512 * x1 + 8192 * x3),
        xmask)
    tmp5 = tl.load(in_ptr0 + (8304 + x0 + 32 * x2 + 512 * x1 + 8192 * x3),
        xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x4, tmp6, xmask)
    tl.store(out_ptr1 + x4, tmp16, xmask)


@triton.jit
def triton_poi_fused_clamp_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (49152, 16384, 1024,
        32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [2,
            2, 2], [1, 1, 1], padding=(1, 1, 1), dilation=(1, 1, 1),
            transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(101776)](buf2, primals_2, 
            101776, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 16, 16, 16, 16), (4096, 256, 16, 1,
            1), torch.float32)
        buf4 = empty_strided_cuda((128, 16, 16, 16, 16), (4096, 256, 16, 1,
            1), torch.int8)
        triton_poi_fused_max_pool3d_with_indices_1[grid(25444)](buf2, buf3,
            buf4, 25444, XBLOCK=256, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((128, 1, 1, 1, 1), (1, 128, 128, 128, 
            128), torch.float32)
        triton_poi_fused_clamp_2[grid(128)](buf3, buf5, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf5, primals_1, primals_3, buf2, buf3, buf4


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
