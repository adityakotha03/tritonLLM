import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_hardswish_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = xindex // 64 % 256
    x2 = xindex // (64 * 256) % 65536
    x4 = xindex // (64 * 256 * 65536)
    tmp0 = tl.load(in_ptr0 + (64 * x1 + 256 * x2 + 16384 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + (64 * x0 + 256 * x2 + 16384 * x3 + 16384 * x1),
        xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 6.0
    tmp4 = tmp2 / tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = -3.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp3
    tmp10 = tmp4 + tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + x3, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128, 16, 16, 16), (32768, 256, 16, 1,
        1))
    assert_size_stride(primals_2, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_3, (64, 1, 1, 1, 1), (1, 1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 32, 32, 32), (65536, 1024, 32, 1,
            1))
        buf1 = empty_strided_cuda((128, 64, 16, 16, 16), (16384, 256, 16, 1,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardswish_0[grid(1048576)](buf0, primals_1,
            buf1, 1048576, XBLOCK=256, num_warps=8, num_stages=1)
        del buf0
        del primals_1
    return buf1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0, input_1):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.bias
        primals_1 = input_0
        add_input = input_1
        output = call([primals_1, primals_2, primals_3])
        return output[0]