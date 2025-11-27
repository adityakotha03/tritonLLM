import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 65536
    xnumel = 1728
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 1728 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64 * x2 + 110592 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_mish_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 1.0
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp6, tmp2)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused_tanh_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = torch.tanh(tmp2)
    tl.store(in_out_ptr0 + x2, tmp3, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 32, 64, 64), (110592, 3465, 1024, 16,
        1))
    assert_size_stride(arg1_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 32, 64, 64), (110592, 1728, 576,
            9, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(65536, 1728)](arg1_1, buf0, 65536,
            1728, XBLOCK=64, YBLOCK=128, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = buf0
        del buf0
        triton_poi_fused_mish_1[grid(1048576)](buf1, arg0_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg0_1
        buf2 = buf1
        del buf1
        triton_poi_fused_tanh_2[grid(1048576)](buf2, arg0_1, 1048576, XBLOCK
            =512, num_warps=8, num_stages=1)
        del arg0_1
    return buf2,


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=
            stride, padding=padding)

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
