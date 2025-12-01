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
def triton_poi_fused_add_mul_rsub_tanh_0(in_out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1973280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 + tmp2
    tmp5 = tmp0 / tmp4
    tmp6 = tl.tanh(tmp5)
    tmp7 = tl.tanh(tmp6)
    tl.store(in_out_ptr0 + x0, tmp7, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(arg1_1, (16, 32, 32, 64, 64), (131072, 4096, 128, 2,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(1, 1, 1),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 30, 62, 62), (1191296, 18645, 621,
            10, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_mul_rsub_tanh_0[grid(1973280)](buf1, 1973280,
            XBLOCK=256, num_warps=4, num_stages=1)
    return buf1, arg0_1, arg1_1


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, input_0):
        arg0_1 = self.conv.weight
        arg1_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]