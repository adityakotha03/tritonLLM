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
def triton_poi_fused_convolution_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 257632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 32
    x0 = xindex % 16384
    x4 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + (x0 + 16448 * x4), tmp7, xmask)


@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1(in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 129024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1536 % 32
    x0 = xindex % 1536
    x4 = xindex // 1536
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 16384 * x4), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.2
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp10, xmask)
    tl.store(out_ptr2 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 64512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1536
    x1 = xindex // 1536 % 3
    x2 = xindex // 4608
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16640 * x1 + 64512 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (32768 + x0 + 16640 * x1 + 64512 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (33280 + x0 + 16640 * x1 + 64512 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (65536 + x0 + 16640 * x1 + 64512 * x2), xmask)
    tmp7 = tl.load(in_ptr0 + (66048 + x0 + 16640 * x1 + 64512 * x2), xmask)
    tmp9 = tl.load(in_ptr0 + (66560 + x0 + 16640 * x1 + 64512 * x2), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp1 > tmp0
    tmp12 = tl.full([1], 1, tl.int8)
    tmp13 = tl.full([1], 0, tl.int8)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp3 > tmp2
    tmp16 = tl.full([1], 2, tl.int8)
    tmp17 = tl.where(tmp15, tmp16, tmp14)
    tmp18 = tmp5 > tmp4
    tmp19 = tl.full([1], 3, tl.int8)
    tmp20 = tl.where(tmp18, tmp19, tmp17)
    tmp21 = tmp7 > tmp6
    tmp22 = tl.full([1], 4, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp9 > tmp8
    tmp25 = tl.full([1], 5, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tl.store(out_ptr0 + x3, tmp10, xmask)
    tl.store(out_ptr1 + x3, tmp26, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 16, 3, 3, 3), (432, 27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (16, 16, 16, 32, 32), (131072, 8192, 512,
        32, 1))
    assert_size_stride(primals_4, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_5, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (16, 32, 17, 33, 33), (303744, 9523, 560, 
            16, 1))
        buf1 = empty_strided_cuda((16, 32, 17, 33, 33), (303744, 9523, 560,
            16, 1), torch.bool)
        buf2 = empty_strided_cuda((16, 32, 17, 33, 33), (303744, 9523, 560,
            16, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_0[grid(257632)](buf0,
            primals_2, buf1, buf2, 257632, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf0
        del primals_2
        buf3 = empty_strided_cuda((16, 32, 17, 33, 33), (303744, 9523, 560,
            16, 1), torch.float32)
        buf4 = empty_strided_cuda((16, 32, 17, 33, 33), (303744, 9523, 560,
            16, 1), torch.bool)
        buf5 = empty_strided_cuda((16, 32, 17, 33, 33), (303744, 9523, 560,
            16, 1), torch.float32)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_1[grid(
            129024)](buf2, primals_4, primals_5, buf3, buf4, buf5, 129024,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf2
        del primals_5
        buf6 = empty_strided_cuda((16, 32, 8, 16, 16), (66560, 2080, 260, 
            16, 1), torch.float32)
        buf7 = empty_strided_cuda((16, 32, 8, 16, 16), (66560, 2080, 260, 
            16, 1), torch.int8)
        triton_poi_fused_max_pool3d_with_indices_2[grid(64512)](buf5, buf6,
            buf7, 64512, XBLOCK=512, num_warps=8, num_stages=1)
        del buf5
    return buf6, primals_1, primals_3, primals_4, buf1, buf3, buf4, buf7


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.multiplier
        primals_3 = input_0
        primals_5 = self.leaky_relu.negative_slope
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
