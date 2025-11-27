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
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 8192 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_sub_tanh_1(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64 % 64
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1 + 8192 * x2), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 - tmp1
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = 0.2
    tmp5 = tmp3 - tmp4
    tl.store(out_ptr0 + x3, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 126, 126), (2097152, 16384, 126,
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(8388608)](buf1, primals_2, 
            8388608, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 128, 64, 64), (524288, 4096, 64, 1),
            torch.float32)
        triton_poi_fused_avg_pool2d_sub_tanh_1[grid(1048576)](buf1, buf2, 
            1048576, XBLOCK=1024, num_warps=4, num_stages=1)
    return buf2, primals_1, primals_3, buf1


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
