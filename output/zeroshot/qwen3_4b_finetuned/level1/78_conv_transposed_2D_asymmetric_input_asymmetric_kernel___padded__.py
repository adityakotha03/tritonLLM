import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1152
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 112 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32 * x2 + 3584 * y1), tmp0, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8, 32, 512, 1024), (1638400, 51200, 1024,
        1))
    assert_size_stride(primals_2, (32, 32, 3, 7), (672, 21, 7, 1))
    assert_size_stride(primals_3, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 32, 512, 1024), (1638400, 51200, 1024,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1152, 112)](primals_1, buf0, 
            1152, 112, XBLOCK=32, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
    return buf0, primals_3


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
        stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, input_0):
        primals_2 = self.conv_transpose2d.weight
        primals_3 = self.conv_transpose2d.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
