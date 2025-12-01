import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64 % 512
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1 + 4096 * x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 5, 9), (1440, 45, 9, 1))
    assert_size_stride(primals_2, (64, 32, 5, 9), (1440, 45, 9, 1))
    assert_size_stride(primals_3, (8, 32, 512, 512), (8388608, 262144, 512, 1
        ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 64, 512, 512), (2097152, 32768, 64, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1024000)](primals_2, primals_1,
            buf0, 1024000, XBLOCK=128, num_warps=4, num_stages=1)
    return buf0, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv2d.weight
        primals_2 = self.conv2d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]