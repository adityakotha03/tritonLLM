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
def triton_poi_fused_add_hardtanh_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 3.0
    tmp5 = tmp2 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = triton_helpers.minimum(tmp7, tmp4)
    tmp9 = tmp8 * tmp3
    tl.store(out_ptr0 + x0, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 32, 16, 16, 16), (131072, 4096, 256,
        16, 1))
    assert_size_stride(primals_4, (128, 64, 32, 16, 16), (2097152, 32768, 1024,
        64, 4))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 32, 16, 16), (2097152, 32768, 
            1024, 64, 4))
        buf1 = empty_strided_cuda((128, 64, 32, 16, 16), (2097152, 32768, 
            1024, 64, 4), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_mul_0[grid(2097152)](buf0, primals_2,
            primals_4, buf1, 2097152, XBLOCK=1024, num_warps=4, num_stages=1)
    return buf1, primals_1, primals_2, primals_3, primals_4, buf0


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0, input_1):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        primals_4 = input_1
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
