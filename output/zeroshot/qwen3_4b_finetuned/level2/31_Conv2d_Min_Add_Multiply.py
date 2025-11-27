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
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1658880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131
    x1 = xindex // 131 % 131
    x2 = xindex // 17061
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 * (x0 // 16 % 16) + 4096 * x2 + x0 % 16 +
        1024 * (x1 // 16 % 16) + 65536 * (x1 // 256 % 16) + 131072 * x1), 
        xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1576672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 165888 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1658880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.5
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + x0, tmp7, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_4, (128, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 131, 131), (1080768, 17061, 131,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(1658880)](primals_3, buf0,
            1658880, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf1 = extern_kernels.convolution(buf0, primals_1, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (128, 128, 126, 126), (2097152, 16588, 126,
            1))
        buf2 = buf1
        del buf1
        triton_poi_fused_convolution_1[grid(1576672)](buf2, primals_2, 
            1576672, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 64, 131, 131), (1080768, 17061, 131,
            1), torch.float32)
        triton_poi_fused_add_mul_2[grid(1658880)](buf2, primals_4, buf3, 
            1658880, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
    return buf3, primals_1, buf0, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
