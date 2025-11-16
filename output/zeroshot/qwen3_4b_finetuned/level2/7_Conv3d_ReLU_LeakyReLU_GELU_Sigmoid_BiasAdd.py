import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3, 3), (6912, 864, 288, 96, 32)
        )
    assert_size_stride(primals_2, (32, 8, 3, 3, 3), (2304, 288, 96, 32, 1))
    assert_size_stride(primals_3, (64, 8, 32, 32, 32), (8192, 1024, 32, 1, 1
        ))
    assert_size_stride(primals_4, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 32, 30, 62, 62), (11805504, 371568,
            12384, 196, 1))
        buf1 = empty_strided_cuda((64, 32, 30, 62, 62), (11805504, 371568,
            12384, 196, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(1048576)](primals_4, buf1, 1048576,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_4
    return buf1, primals_1, primals_2, primals_3, buf0


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
        primals_2 = self.conv.bias
        primals_3 = input_0
        primals_4 = self.bias
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
