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


@triton.jit
def triton_per_fused__to_copy_convolution_0(in_out_ptr0, in_ptr0, in_ptr1,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 384
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    r3 = rindex // 256
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * r1 + 4096 * r3), xmask,
        other=float('-inf'))
    tmp3 = tl.load(in_ptr1 + (24 * r1 + 49152 * r3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0 + 128 * r1 + 4096 * r3), tmp1, xmask)
    tl.store(out_ptr0 + (r1 + 1024 * x0), tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8, 48, 3, 3, 3), (108, 24, 9, 3, 1))
    assert_size_stride(primals_2, (24, 48, 3, 3, 3), (108, 3, 9, 3, 1))
    assert_size_stride(primals_3, (8, 24, 102, 102, 102), (41616, 102, 102,
        1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 48, 102, 102, 102), (41616, 864, 41, 4,
            1), torch.float32)
        buf1 = buf0
        del buf0
        buf3 = empty_strided_cuda((8, 24, 102, 102, 102), (41616, 1728, 41, 4,
            1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__to_copy_convolution_0[grid(384)](buf1, primals_2,
            primals_3, buf3, 384, 128, XBLOCK=64, num_warps=4, num_stages=1)
        del primals_3
        del primals_2
    return buf3, primals_1, buf1


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
        primals_2 = self.conv_transpose3d.weight
        primals_1 = self.conv_transpose3d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
