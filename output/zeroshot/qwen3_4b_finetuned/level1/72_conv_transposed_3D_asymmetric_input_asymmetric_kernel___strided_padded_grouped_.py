import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 691200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 32, 3, 5, 7), (240, 8, 3, 6, 1))
    assert_size_stride(primals_2, (8, 32, 12, 24, 48), (36864, 12, 4, 16, 1))
    assert_size_stride(primals_3, (32,), (1,))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32, 32, 3, 5, 7), (1280, 1280, 480, 96, 16))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_2, (8, 
            32, 13, 27, 52), (53760, 12, 12, 5), (0, 0, 0, 0, 0), 0), primals_1,
            stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
            transposed=True, output_padding=(1, 1, 1), groups=1,
            bias=None)
        assert_size_stride(buf0, (8, 32, 14, 26, 51), (3096864, 96768, 24064, 
            960, 16))
        buf1 = empty_strided_cuda((8, 32, 15, 29, 54), (3549792, 110976, 
            3840, 128, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(691200)](buf0, buf1, 691200,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2, primals_3, primals_4, primals_5


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple of ints): Size of the convolution kernel in the form (kernel_size_depth, kernel_size_height, kernel_size_width).
        stride (tuple of ints, optional): Stride of the convolution in the form (stride_depth, stride_height, stride_width). Defaults to (1, 1, 1).
        padding (tuple of ints, optional): Padding applied to the input in the form (padding_depth, padding_height, padding_width). Defaults to (0, 0, 0).
        output_padding (tuple of ints, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_3 = self.conv_transpose3d.bias
        primals_4 = self.conv_transpose3d.weight_ref
        primals_5 = input_0
        output = call([primals_1, primals_5, primals_3, primals_4, primals_2])
        return output[0]
