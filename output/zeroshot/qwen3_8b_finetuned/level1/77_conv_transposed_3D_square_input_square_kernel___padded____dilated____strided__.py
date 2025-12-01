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
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex // 108 % 32
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex // 108 % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + x2, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 32, 16, 32, 32), (163840, 5120, 160,
        5, 1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 32, 3, 3, 3), (864, 27, 9, 3, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(55296)](primals_1, buf0, 55296, XBLOCK=512,
            num_warps=8, num_stages=1)
        del primals_1
        buf1 = extern_kernels.convolution(primals_2, buf0, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(2, 2, 2), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (16, 64, 8, 14, 14), (98304, 1536, 192, 14,
            1))
        buf2 = buf1
        del buf1
        triton_poi_fused_convolution_1[grid(98304)](buf2, primals_3, 98304,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
    return buf2, primals_2, buf0


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square kernel, so only one value needed).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_3 = self.conv_transpose3d.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]