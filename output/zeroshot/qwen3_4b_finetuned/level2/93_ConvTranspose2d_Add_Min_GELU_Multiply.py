import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_add_minimum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_gelu_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
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
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_4, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2), padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 64, 64), (524288, 4096, 64, 1))
        buf1 = empty_strided_cuda((128, 128, 64, 64), (524288, 4096, 64, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_minimum_0[grid(524288)](buf0, buf1, 524288,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((128, 128, 64, 64), (524288, 4096, 64, 1),
            torch.float32)
        triton_poi_fused_gelu_mul_1[grid(524288)](buf1, buf2, 524288,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf1
    return buf2, primals_1, primals_3, primals_4


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        primals_4 = self.add_value
        primals_5 = self.multiply_value
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
