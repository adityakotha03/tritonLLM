import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 64
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 8
    y1 = yindex // 8
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8 * x2 + 128 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 64
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 8
    y1 = yindex // 8
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8 * x2 + 128 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.load(in_ptr2 + (y0 + 8 * x2 + 128 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2 + 16 * y3), tmp4, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 5, 5), (4800, 1536, 800, 160, 
        32))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 64, 64, 64), (131072, 4096, 64, 
        64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64, 64, 64, 64), (4096, 64, 1, 1, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(64, 16)](primals_3, primals_1, buf0, 
            64, 16, XBLOCK=16, YBLOCK=64, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((16, 64, 1, 1, 1), (64, 1, 16, 16, 16), 
            torch.float32)
        triton_poi_fused_clone_1[grid(64, 16)](primals_3, primals_1,
            primals_2, buf1, 64, 16, XBLOCK=16, YBLOCK=64, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
    return reinterpret_tensor(buf0, (16, 64, 68, 68, 68), (104065472, 1600064
        , 1536, 24), 0), reinterpret_tensor(buf1, (16, 64, 68, 68, 68), (
        104065472, 1600064, 1536, 24), 0), primals_3


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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
        stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_2 = self.conv_transpose3d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
