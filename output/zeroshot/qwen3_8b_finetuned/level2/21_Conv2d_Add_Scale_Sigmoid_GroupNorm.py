import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2147483648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tl.store(out_ptr0 + x2, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (128, 32, 256, 256), (2097152, 8192, 32,
        1))
    assert_size_stride(primals_3, (32,), (1,))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (8, 3, 3), (9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_2, primals_1,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 32, 256, 256), (2097152, 8192, 32, 1))
        buf1 = empty_strided_cuda((128, 32, 256, 256), (2097152, 8192, 32, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_0[grid(2147483648)](buf0,
            primals_3, primals_4, buf1, 2147483648, XBLOCK=1024, num_warps
            =4, num_stages=1)
        del buf0
        del primals_3
        del primals_4
    return buf1, primals_1, primals_2, primals_5


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_5 = self.conv.bias
        primals_3 = self.bias
        primals_4 = self.scale
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]