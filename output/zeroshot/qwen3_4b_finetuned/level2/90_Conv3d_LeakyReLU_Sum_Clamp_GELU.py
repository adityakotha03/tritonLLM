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
def triton_poi_fused_convolution_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 13421776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 64
    x0 = xindex % 4096
    x4 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp0 * tmp4
    tmp6 = tl.where(tmp2, tmp0, tmp5)
    tmp7 = tmp6 + tmp3
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tl.store(out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr1 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused_add_clamp_ge_le_logical_and_1(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 13421776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = -1.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp6 >= tmp3
    tmp8 = tmp6 <= tmp5
    tmp9 = tmp7 & tmp8
    tl.store(out_ptr0 + x0, tmp6, xmask)
    tl.store(out_ptr1 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 13421776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 16, 64, 64), (524288, 65536, 4096,
        64, 1))
    assert_size_stride(primals_4, (64, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 14, 62, 62), (3528640, 55135, 
            3945, 62, 1))
        buf1 = empty_strided_cuda((128, 64, 14, 62, 62), (3528640, 55135, 
            3945, 62, 1), torch.bool)
        buf2 = empty_strided_cuda((128, 64, 14, 62, 62), (3528640, 55135, 
            3945, 62, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_0[grid(13421776)](buf0,
            primals_2, buf1, buf2, 13421776, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del buf0
        del primals_2
        buf3 = empty_strided_cuda((128, 64, 14, 62, 62), (3528640, 55135, 
            3945, 62, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 64, 14, 62, 62), (3528640, 55135, 
            3945, 62, 1), torch.bool)
        triton_poi_fused_add_clamp_ge_le_logical_and_1[grid(13421776)](buf2,
            primals_4, buf3, buf4, 13421776, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf2
        del primals_4
        buf5 = buf1
        del buf1
        triton_poi_fused_gelu_2[grid(13421776)](buf3, buf5, 13421776,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
    return buf5, primals_1, primals_3, buf4


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, input_0):
        primals_4 = self.sum_tensor
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
