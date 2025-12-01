import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr1, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 5, 7), (2240, 35, 7, 1))
    assert_size_stride(primals_2, (8, 64, 512, 256), (8388608, 131072, 256,
        1))
    assert_size_stride(primals_3, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1048576, 896), (896, 1), torch.float32)
        buf1 = torch.ops.aten._convolution.convolution(primals_2, primals_1,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        del primals_1
        buf2 = buf1[0]
        del buf1
        buf3 = buf1[1]
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1048576)](buf2, primals_3, 
            1048576, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
    return buf2, primals_2, buf3


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv2d.weight
        primals_3 = self.conv2d.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]