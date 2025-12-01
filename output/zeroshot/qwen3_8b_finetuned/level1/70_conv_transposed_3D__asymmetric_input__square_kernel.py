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
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x2 = xindex
    x4 = xindex // 432
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + (x4 + 1152 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 + tmp0
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (24, 48, 3, 3, 3), (1152, 24, 8, 2, 1))
    assert_size_stride(primals_2, (24,), (1,))
    assert_size_stride(primals_3, (8, 48, 96, 96, 96), (4608, 96, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 24, 3, 3, 3), (216, 9, 3, 1, 1), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(864)](primals_2, primals_3,
            primals_1, buf0, 864, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf1 = extern_kernels.convolution(buf0, primals_1, stride=(1, 1, 1),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 24, 98, 98, 98), (23528, 98, 98, 1, 1))
    return buf1, primals_3, buf0


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_2 = self.conv_transpose3d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]