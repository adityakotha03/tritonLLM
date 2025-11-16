import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__unsafe_index_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x3 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 1280 * x3), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 3840
    x1 = xindex // 1280 % 3
    x4 = xindex % 3840
    x2 = xindex // 3840
    x6 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x4 + 3840 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 3 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x6, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 5), (960, 15, 5, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 64, 128, 256), (204800, 3200, 256, 1
        ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 64, 5, 3), (960, 1, 192, 64), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_0[grid(16384)](primals_1, buf0, 
            16384, XBLOCK=256, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((256, 128, 3, 5), (1920, 15, 64, 1), torch
            .float32)
        triton_poi_fused__unsafe_index_1[grid(49152)](primals_2, buf1, 
            49152, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = extern_kernels.convolution(reinterpret_tensor(primals_3, (
            256, 64, 128, 256), (204800, 3200, 1600, 6400), 0), buf0, stride=(
            1, 1), padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (256, 128, 132, 260), (424160, 3200, 3200,
            1280))
        buf3 = buf2
        del buf2
        triton_poi_fused_convolution_2[grid(122880)](buf3, buf1, 122880,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
    return buf3, primals_1, reinterpret_tensor(primals_3, (256, 64, 128, 256
        ), (204800, 3200, 1600, 6400), 0)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
        stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple
        = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv_transpose2d.weight
        primals_2 = self.conv_transpose2d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
