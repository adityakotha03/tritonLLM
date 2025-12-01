import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 132051456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + (x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tl.load(in_ptr0 + tl.broadcast_to(x2, [XBLOCK]), xmask)
    tmp5 = tmp4 + tmp3
    tl.store(out_ptr0 + x2, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 3, 7), (672, 21, 7, 1))
    assert_size_stride(primals_2, (32, 32, 513, 1027), (16777216, 524288, 
        3136, 3))
    assert_size_stride(primals_3, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 513, 1027), (16777216, 524288, 
            3136, 3), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(132051456)](primals_3,
            primals_1, primals_2, buf0, 132051456, XBLOCK=128, num_warps=8,
            num_stages=1)
    return buf0, primals_1, primals_2, primals_3


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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv_transpose2d.weight
        primals_3 = self.conv_transpose2d.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]