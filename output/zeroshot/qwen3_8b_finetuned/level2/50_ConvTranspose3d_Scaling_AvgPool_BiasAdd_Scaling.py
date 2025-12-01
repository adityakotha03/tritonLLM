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
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 21070400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 20132160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 20132160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 16, 3, 3, 3), (162, 101, 303, 101, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (196608, 65536, 4096,
        128, 4))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 31, 65, 65), (65536, 4096, 131,
            20, 1))
        buf1 = empty_strided_cuda((128, 16, 31, 65, 65), (65536, 4096, 131,
            20, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(21070400)](buf0, primals_2, buf1, 
            21070400, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = extern_kernels.avg_pool3d(buf1, kernel_size=(2, 2, 2), stride
            =(2, 2, 2), padding=(0, 0, 0), count_include_pad=False)
        assert_size_stride(buf2, (128, 16, 30, 64, 64), (65536, 4096, 131,
            20, 1))
        buf3 = empty_strided_cuda((128, 16, 30, 64, 64), (65536, 4096, 131,
            20, 1), torch.float32)
        triton_poi_fused_mul_1[grid(20132160)](buf2, primals_4, buf3, 
            20132160, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((128, 16, 30, 64, 64), (65536, 4096, 131,
            20, 1), torch.float32)
        triton_poi_fused_add_1[grid(20132160)](buf3, primals_5, buf4, 
            20132160, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
    return buf4, primals_1, primals_3, buf0, buf1, buf2, buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.scale1
        primals_5 = self.bias
        primals_4 = self.scale2
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]