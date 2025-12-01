import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 13007168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512 % 511
    x2 = xindex // 261728
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1 + 261728 * x2), xmask)
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = xindex // 24
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24 * x1), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 13007168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512 % 511
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8, 8, 3, 1), (24, 3, 1, 1))
    assert_size_stride(primals_2, (64, 8, 512, 512), (2097152, 262144, 512,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 8, 511, 511), (2095328, 262144, 512,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(13007168)](primals_2, buf0, 
            13007168, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((8, 8, 3, 1), (24, 3, 1, 1), torch.float32)
        triton_poi_fused_convolution_1[grid(24576)](primals_1, buf1, 24576,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf2 = buf0
        del buf0
        triton_poi_fused_convolution_2[grid(13007168)](buf2, buf1, 13007168,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
    return buf2, primals_1


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv2d.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]