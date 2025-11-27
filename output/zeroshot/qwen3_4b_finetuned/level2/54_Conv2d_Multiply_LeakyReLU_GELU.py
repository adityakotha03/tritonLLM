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
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tl.store(out_ptr1 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
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
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64, 64, 256, 256), (4194304, 65536, 256,
        1))
    assert_size_stride(primals_4, (64, 1, 1), (1, 1, 1))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 64, 254, 254), (4194304, 65536, 256, 1))
        buf1 = empty_strided_cuda((64, 1, 1), (1, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(64)](primals_4, primals_5, buf1, 64,
            XBLOCK=64, num_warps=1, num_stages=1)
        del primals_4
        del primals_5
        buf2 = empty_strided_cuda((64,), (1,), torch.bool)
        buf3 = empty_strided_cuda((64,), (1,), torch.float32)
        triton_poi_fused_leaky_relu_1[grid(64)](buf0, buf1, buf2, buf3, 64,
            XBLOCK=64, num_warps=1, num_stages=1)
        buf4 = empty_strided_cuda((64, 64, 254, 254), (4194304, 65536, 256,
            1), torch.float32)
        triton_poi_fused_gelu_2[grid(1048576)](buf3, buf4, 1048576, XBLOCK=
            512, num_warps=8, num_stages=1)
        del buf3
    return buf4, primals_1, primals_3, buf0, buf1, buf2


class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.multiplier
        primals_3 = input_0
        primals_5 = self.leaky_relu.weight
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
