import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_convolution_0(in_out_ptr0, in_ptr0, in_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 164896 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x3, xmask)
    tmp3 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_hardswish_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4 > 0.0
    tmp6 = tmp4 <= 0.0
    tmp7 = tmp6 & tmp5
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp4 - tmp10
    tmp12 = 3.0
    tmp13 = tmp11 * tmp12
    tmp14 = tl.where(tmp7, tmp4, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp5, tmp4, tmp15)
    tmp17 = tl.where(tmp6, tmp14, tmp16)
    tl.store(out_ptr0 + x0, tmp17, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(arg1_1, (128, 64, 16, 16, 16), (16384, 256, 16, 1, 1))
    assert_size_stride(arg2_1, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 16, 16, 16), (16384, 256, 16, 1, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_0[grid(1179648)](buf0, arg2_1,
            arg1_1, 1179648, XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        del arg2_1
        buf1 = empty_strided_cuda((128, 64, 16, 16, 16), (16384, 256, 16, 1,
            1), torch.float32)
        triton_poi_fused_hardswish_mul_1[grid(1179648)](arg0_1, buf0, buf1,
            1179648, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
        output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0, input_1):
        arg0_1 = self.conv_transpose.weight
        arg2_1 = self.bias
        output = call([input_0, input_1, arg0_1])
        return output[0]
