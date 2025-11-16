import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2048 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 3, 3, 3), (81, 1, 27, 9, 3))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (4, 32, 32, 64, 128), (8192, 256, 64, 16, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (4, 
            32, 32, 64, 128), (131072, 4096, 128, 2, 1), 0), primals_1, stride
            =(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=
            True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 34, 66, 130), (1485632, 46400, 128,
            2, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(2097152)](buf1, primals_2, 2097152
            , XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
    return reinterpret_tensor(buf1, (4, 32, 34, 66, 130), (1485632, 46400, 128
        , 2, 1), 0), primals_1, reinterpret_tensor(primals_3, (4, 32, 32, 64,
        128), (131072, 4096, 128, 2, 1), 0)


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_2 = self.conv_transpose3d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
