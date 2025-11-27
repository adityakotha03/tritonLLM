import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = xindex // 131072
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 327680 * x1), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 131072 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3), (96, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 131072), (4192672, 131072, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 131072), (819200, 131072, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(819200)](primals_3, buf0, 
            819200, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
        buf1 = extern_kernels.convolution(buf0, primals_1, stride=(2,),
            padding=(1,), dilation=(2,), transposed=True, output_padding=(
            0,), groups=1, bias=None)
        assert_size_stride(buf1, (16, 64, 131072), (819200, 131072, 1))
        buf2 = buf1
        del buf1
        triton_poi_fused_convolution_1[grid(5242880)](buf2, primals_2, 
            5242880, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
    return buf2, primals_1, buf0


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv1d_transpose.weight
        primals_2 = self.conv1d_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
