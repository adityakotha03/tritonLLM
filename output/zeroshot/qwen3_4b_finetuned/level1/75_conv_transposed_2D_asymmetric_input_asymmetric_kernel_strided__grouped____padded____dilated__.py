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
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (y3 + 128 * x2 + 4096 * y0), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 32 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (y3 + 128 * x2 + 4096 * y0), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 32 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (y3 + 128 * x2 + 4096 * y0), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 32 * y3), tmp0, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 5), (480, 15, 5, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 128, 256), (1048576, 32768, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 32, 3, 5), (480, 15, 5, 1), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(128, 32)](primals_1, buf0, 128, 32, XBLOCK=
            32, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((64, 32, 3, 5), (480, 15, 5, 1), torch
            .float32)
        triton_poi_fused_0[grid(128, 32)](primals_2, buf1, 128, 32, XBLOCK=
            32, YBLOCK=64, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((64, 32, 3, 5), (480, 15, 5, 1), torch
            .float32)
        triton_poi_fused_1[grid(128, 32)](primals_3, buf2, 128, 32, XBLOCK=
            32, YBLOCK=64, num_warps=4, num_stages=1)
        buf3 = extern_kernels.convolution(buf2, buf0, stride=(2, 3),
            padding=(1, 2), dilation=(2, 1), transposed=True,
            output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf3, (16, 64, 141, 272), (1188736, 18576, 84, 1))
        buf4 = extern_kernels.convolution(buf3, buf1, stride=(2, 3),
            padding=(1, 2), dilation=(2, 1), transposed=True,
            output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf4, (16, 64, 141, 272), (1188736, 18576, 84, 1))
        buf5 = buf4
        del buf4
        triton_poi_fused_2[grid(128, 32)](primals_3, buf5, 128, 32, XBLOCK=
            32, YBLOCK=64, num_warps=4, num_stages=1)
    return buf5, primals_3, reinterpret_tensor(buf0, (64, 3, 5, 32), (480, 
        15, 5, 1), 0), reinterpret_tensor(buf1, (64, 1, 1, 1), (1, 1, 1, 1),
        0), reinterpret_tensor(buf2, (16, 32, 128, 256), (1048576, 32768, 
        256, 1), 0)


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
        stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple =
        (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv_transpose2d.weight
        primals_2 = self.conv_transpose2d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
