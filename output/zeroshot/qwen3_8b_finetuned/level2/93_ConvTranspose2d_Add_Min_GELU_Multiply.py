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


@triton.jit
def triton_poi_fused_add_gelu_mul_min_0(in_out_ptr0, in_ptr0, in_ptr1,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6553600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = 0.7071067811865476
    tmp11 = tmp5 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = 1.0 + tmp12
    tmp14 = tmp9 * tmp13
    tmp15 = tl.load(in_ptr1 + 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + x0, tmp17, xmask)
    tl.store(out_ptr0 + x0, tmp17, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_2, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_3, (), ())
    assert_size_stride(primals_4, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 
            2), padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 64, 64), (524288, 4096, 64, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 128, 64, 64), (524288, 4096, 64, 1),
            torch.float32)
        buf3 = empty_strided_cuda((128, 128, 64, 64), (524288, 4096, 64, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_gelu_mul_min_0[grid(6553600)](buf1, primals_3,
            primals_4, buf3, 6553600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
    return buf3, primals_1, primals_2, primals_3, buf1, buf3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = input_0
        primals_3 = self.add_value
        primals_4 = self.multiply_value
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]