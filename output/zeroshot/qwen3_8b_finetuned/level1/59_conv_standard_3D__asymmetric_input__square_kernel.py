import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 27
    x2 = xindex // 65536
    x1 = xindex // 27 % 65536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 27 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 27), (27, 1))
    assert_size_stride(primals_2, (16, 3, 256, 256, 10), (1966080, 655360,
        2560, 10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1048576, 27), (27, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_2, (1048576, 27), (27,
            1), 0), reinterpret_tensor(primals_1, (27, 64), (1, 27), 0),
            out=buf0)
        del primals_1
        buf1 = reinterpret_tensor(buf0, (16, 64, 256, 256, 10), (65536, 1024,
            4, 16, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1048576)](primals_2, buf1,
            buf1, 1048576, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = buf1
        del buf1
        triton_poi_fused_convolution_1[grid(65536)](buf2, buf2, 65536,
            XBLOCK=256, num_warps=2, num_stages=1)
    return buf2, reinterpret_tensor(primals_1, (64, 27), (1, 64), 0
        ), reinterpret_tensor(buf2, (16, 64, 256, 256, 10), (65536, 1024, 4,
        16, 1), 0)


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv3d.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]