import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_tanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + x2, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32, 64, 256, 256), (4194304, 65536, 256, 
        1))
    assert_size_stride(primals_2, (64, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (32, 64, 258, 258), (42377664, 663072, 258, 
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(524288)](buf1, primals_3, 524288, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_3
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_1 = input_0
        primals_3 = self.bias
        output = call([primals_1, primals_2, primals_3])
        return output[0]
