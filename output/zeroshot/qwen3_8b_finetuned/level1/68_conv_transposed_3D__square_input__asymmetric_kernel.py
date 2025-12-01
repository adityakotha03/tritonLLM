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
def triton_poi_fused_add_convolution_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = xindex // 64 % 2048
    x2 = xindex // 131072
    x4 = xindex % 2048
    x5 = xindex // 2048
    tmp0 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + 2560 * x2 + 40960 * x1 + 163840 * x0),
        xmask)
    tmp3 = tl.load(in_ptr1 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (12 + 8 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr1 + (20 + 8 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp6
    tmp7 = tmp5 + tmp9
    tmp8 = tmp7 + tmp9
    tmp10 = tmp8 + tmp6
    tmp11 = tmp10 + tmp3
    tl.store(out_ptr0 + x3, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 5, 5), (2560, 80, 26, 5, 1))
    assert_size_stride(primals_2, (16, 32, 64, 64, 64), (786432, 24576, 384,
        6, 1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 64, 64, 64), (16777216, 262144, 
            4096, 64, 1))
        buf1 = empty_strided_cuda((16, 64, 64, 64, 64), (16777216, 262144, 
            4096, 64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_0[grid(2097152)](primals_3,
            buf0, primals_1, buf1, 2097152, XBLOCK=128, num_warps=4,
            num_stages=1)
        del primals_1
        del buf0
    return buf1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_3 = self.conv_transpose3d.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]