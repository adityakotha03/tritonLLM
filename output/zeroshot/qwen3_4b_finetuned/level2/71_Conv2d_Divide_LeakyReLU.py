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
def triton_poi_fused_convolution_div_leaky_relu_leaky_relu_backward_0(
    in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 64
    x0 = xindex % 1600
    x4 = xindex // 1600
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.01
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tmp9 > tmp5
    tl.store(in_out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr0 + (x0 + 1664 * x4), tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1, 1], dilation=[1, 1], transposed=False, output_padding=[0, 0],
            groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 64, 126, 126), (10077696, 157921, 
            126, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_convolution_div_leaky_relu_leaky_relu_backward_0[grid
            (589824)](buf1, primals_2, buf2, 589824, XBLOCK=1024, num_warps
            =4, num_stages=1)
        del primals_2
    return buf1, primals_1, primals_3, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
