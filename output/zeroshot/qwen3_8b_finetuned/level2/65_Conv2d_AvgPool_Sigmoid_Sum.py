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
def triton_poi_fused_add_avg_pool2d_convolution_mul_sigmoid_sum_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 68797248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tl.load(in_ptr1 + 0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_3, (128, 8, 384, 384), (1048576, 131072, 32,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 382, 382), (8149856, 127310, 33,
            1))
        del primals_1
        buf1 = extern_kernels.avg_pool2d(buf0, kernel_size=(4, 4),
            stride=(4, 4), padding=(0, 0), ceil_mode=False, count_include_pad
            =False)
        assert_size_stride(buf1, (128, 64, 92, 92), (53824, 835, 9, 1))
        del buf0
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((128,), (1,), torch.float32)
        buf4 = buf3
        del buf3
        get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_convolution_mul_sigmoid_sum_0[grid(68797248)](buf2,
            primals_2, buf4, 68797248, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
    return buf4, primals_3, buf2


class ModelNew(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]