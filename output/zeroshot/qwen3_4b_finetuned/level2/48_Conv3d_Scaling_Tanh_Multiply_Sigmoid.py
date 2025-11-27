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
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1843200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_sigmoid_tanh_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1843200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tmp4 = libdevice.tanh(tmp2)
    tmp5 = tmp4 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 64, 64), (196608, 65536, 
        4096, 64, 1))
    assert_size_stride(primals_4, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_5, (16, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 16, 64, 64), (1843200, 115200,
            7200, 1152, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(1843200)](primals_4, buf0, 1843200,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
        buf1 = torch.ops.aten.convolution.default(primals_3, primals_1, [0,
            0, 0], dilation=[1, 1, 1], transposed=False, output_padding=[0,
            0, 0], groups=1, bias=None)
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((128, 16, 16, 64, 64), (1843200, 115200,
            7200, 1152, 1), torch.float32)
        triton_poi_fused_mul_sigmoid_tanh_1[grid(1843200)](buf2, primals_2,
            primals_5, buf3, 1843200, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf2
        del primals_2
        del primals_5
    return buf3, primals_1, primals_3, buf0


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_4 = self.scaling_factor
        primals_5 = self.bias
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
