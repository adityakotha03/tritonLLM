import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096 % 64
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_mul_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_gelu_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.5
    tmp4 = tmp0 * tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 1.0
    tmp10 = tmp0 * tmp9
    tmp11 = tmp10 * tmp6
    tmp12 = tmp8 * tmp6
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = 0.5
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + x0, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64, 64, 256, 256), (4096, 64, 16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64, 256, 256), (4096, 1, 16, 1), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(16777216)](primals_3, primals_1,
            buf0, 16777216, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((64, 64, 256, 256), (4096, 1, 16, 1), torch
            .float32)
        triton_poi_fused_leaky_relu_1[grid(1048576)](buf0, buf1, 1048576,
            XBLOCK=512, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((64, 64, 256, 256), (4096, 1, 16, 1), torch
            .float32)
        triton_poi_fused_mul_2[grid(1048576)](buf2, primals_2, 1048576,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((64, 64, 256, 256), (4096, 1, 16, 1), torch
            .float32)
        triton_poi_fused_gelu_3[grid(1048576)](buf2, buf3, 1048576,
            XBLOCK=512, num_warps=4, num_stages=1)
        del buf2
    return buf3, reinterpret_tensor(buf1, (64, 64, 256, 256), (4096, 1, 16, 1
        ), 0), reinterpret_tensor(primals_3, (64, 64, 256, 256), (4096, 1,
        16, 1), 0)


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
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
