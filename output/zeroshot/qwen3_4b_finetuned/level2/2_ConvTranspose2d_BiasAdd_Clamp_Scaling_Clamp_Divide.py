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
def triton_poi_fused_add_clamp_div_mul_0(in_out_ptr0, in_ptr0, in_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 129024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 1.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp5)
    tmp12 = triton_helpers.minimum(tmp11, tmp7)
    tmp13 = tmp12 / tmp9
    tl.store(in_out_ptr0 + x2, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_4, (64, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], dilation=[1, 1], transposed=True, output_padding=[1, 1],
            groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128, 1))
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_0[grid(129024)](buf2, primals_2,
            primals_4, 129024, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        del primals_4
    return buf2, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
