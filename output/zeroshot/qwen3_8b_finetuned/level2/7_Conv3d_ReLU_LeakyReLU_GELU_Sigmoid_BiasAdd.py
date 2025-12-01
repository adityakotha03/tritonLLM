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
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4 > tmp3, tmp4, tmp6)
    tl.store(out_ptr0 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = 1.0
    tmp5 = tmp2 * tmp3
    tmp6 = 1.4142135623730951
    tmp7 = tmp2 / tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = tmp4 + tmp8
    tmp10 = tmp5 * tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_sigmoid_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr0 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused_add_sigmoid_4(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr0 + x2, tmp7, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_3, (32, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_2, (64,
            8, 32, 64, 64), (2097152, 262144, 4096, 64, 1), 0), primals_1,
            stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
            transposed=False, output_padding=(0, 0, 0), groups=1,
            bias=None)
        assert_size_stride(buf0, (64, 32, 30, 62, 62), (5881504, 183856, 6128,
            99, 1))
        buf1 = empty_strided_cuda((64, 32, 30, 62, 62), (5881504, 183856, 
            6128, 99, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_relu_0[grid(536870912)](buf0, buf1, 536870912,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((64, 32, 30, 62, 62), (5881504, 183856, 
            6128, 99, 1), torch.float32)
        triton_poi_fused_leaky_relu_1[grid(536870912)](buf1, buf2, 
            536870912, XBLOCK=256, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((64, 32, 30, 62, 62), (5881504, 183856, 
            6128, 99, 1), torch.float32)
        triton_poi_fused_gelu_2[grid(536870912)](buf2, buf3, 536870912,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((64, 32, 30, 62, 62), (5881504, 183856, 
            6128, 99, 1), torch.float32)
        triton_poi_fused_sigmoid_3[grid(536870912)](buf3, buf4, 536870912,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((64, 32, 30, 62, 62), (5881504, 183856, 
            6128, 99, 1), torch.float32)
        triton_poi_fused_add_sigmoid_4[grid(536870912)](buf4, primals_3, buf5,
            536870912, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
        del primals_3
    return buf5, primals_1, primals_2, buf0, buf1, buf2, buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]