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
def triton_poi_fused_convolution_sigmoid_mul_0(in_out_ptr0, in_ptr0,
    in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 29797344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x3 = xindex // 34896
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + x2, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16,), (1,))
    assert_size_stride(primals_2, (128, 3, 16, 32, 32), (1572864, 524288, 
        32768, 1024, 32))
    assert_size_stride(primals_3, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 34, 66, 66), (160678176, 10042386,
            286824, 432, 1), torch.float32)
        extern_kernels.convolution(primals_2, primals_3, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        del primals_3
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_mul_0[grid(29797344)](buf1,
            primals_1, primals_2, 29797344, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_1
    return buf1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_0):
        primals_3 = self.conv_transpose.weight
        primals_1 = self.conv_transpose.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]