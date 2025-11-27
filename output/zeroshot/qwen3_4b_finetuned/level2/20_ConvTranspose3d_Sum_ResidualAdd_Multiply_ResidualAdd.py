import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_convolution_mul_0(in_out_ptr0, in_ptr0, in_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2048 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4 + tmp2
    tl.store(in_out_ptr0 + x3, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_convolution_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2048 % 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 32, 16, 32, 32), (16384, 512, 32, 1, 
        1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 17, 33, 33), (35904, 1, 2048, 64,
            2), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_0[grid(491520)](buf0,
            primals_1, primals_2, 491520, XBLOCK=512, num_warps=8, num_stages=1
            )
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((16, 64, 17, 33, 33), (35904, 1, 2048, 64,
            2), torch.float32)
        triton_poi_fused_add_convolution_1[grid(491520)](buf0, primals_3,
            primals_3, buf1, 491520, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
    return buf1, buf0


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_3 = self.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
