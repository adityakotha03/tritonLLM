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
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 18737408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 11236 % 2048
    x2 = xindex // 224768
    tmp0 = x3
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 11236, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 2048 * x2), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tl.full([1], 224768, tl.int64)
    tmp11 = tl.load(in_ptr1 + (x1 + 2048 * x2), tmp8 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + x3, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (48, 48, 3, 3, 3), (1036, 216, 72, 24, 8)
        )
    assert_size_stride(primals_2, (48,), (1,))
    assert_size_stride(primals_3, (8, 48, 64, 64, 64), (196608, 4096, 64, 
        64, 1))
    assert_size_stride(primals_4, (48,), (1,))
    assert_size_stride(primals_5, (8, 48, 64, 64, 64), (196608, 4096, 64, 
        64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((48, 48, 3, 3, 3), (1036, 216, 72, 24, 8
            ), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(18737408)](primals_1, primals_2, buf0,
            18737408, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_2
        buf1 = extern_kernels.convolution(primals_3, reinterpret_tensor(buf0,
            (8, 48, 64, 64, 64), (196608, 4096, 64, 64, 1), 0), stride=(1, 1,
            1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 48, 192, 192, 192), (147456, 3072, 49152
            , 256, 1))
        buf2 = empty_strided_cuda((8, 48, 192, 192, 192), (147456, 3072, 49152
            , 256, 1), torch.float32)
        triton_poi_fused_clone_0[grid(18737408)](primals_4, primals_5, buf2,
            18737408, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_5
    return buf1, primals_1, primals_3, primals_4, reinterpret_tensor(buf0,
        (8, 48, 64, 64, 64), (196608, 4096, 64, 64, 1), 0), buf2


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_2 = self.conv_transpose3d.bias
        primals_4 = self.conv_transpose3d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_4, primals_3])
        return output[0]
