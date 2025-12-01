import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 16, 16, 32, 32), (33554432, 2097152, 128,
        4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(arg0_1, empty_strided_cuda
            (16, 3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1,
            1, 1), transposed=True, output_padding=(0, 0, 0), groups=1,
            bias=None)
        assert_size_stride(buf0, (128, 16, 16, 32, 32), (33554432, 2097152,
            128, 4, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(33554432)](buf1, 33554432, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf1,


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]