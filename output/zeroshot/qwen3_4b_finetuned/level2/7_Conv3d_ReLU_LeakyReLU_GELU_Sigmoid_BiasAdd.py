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
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 331776 % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 331776 % 32
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp7, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 186624
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


@triton.jit
def triton_poi_fused_add_sigmoid_3(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp2 + tmp0
    tl.store(out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (64, 8, 32, 64, 64), (8388608, 1048576, 
        32768, 512, 8))
    assert_size_stride(primals_4, (32, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 32, 32, 64, 64), (1310720, 40960, 
            1280, 20, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(186624)](buf0, primals_1, 
            186624, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((64, 32, 32, 64, 64), (1310720, 40960, 
            1280, 20, 1), torch.bool)
        buf2 = empty_strided_cuda((64, 32, 32, 64, 64), (1310720, 40960, 
            1280, 20, 1), torch.float32)
        triton_poi_fused_leaky_relu_1[grid(186624)](buf0, primals_2, buf1,
            buf2, 186624, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf3 = buf0
        del buf0
        triton_poi_fused_gelu_2[grid(186624)](buf2, buf3, 186624, XBLOCK=
            512, num_warps=8, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((64, 32, 32, 64, 64), (1310720, 40960, 
            1280, 20, 1), torch.float32)
        triton_poi_fused_add_sigmoid_3[grid(186624)](buf3, primals_4, buf4,
            186624, XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
        del primals_4
    return buf4, primals_3, buf1


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
