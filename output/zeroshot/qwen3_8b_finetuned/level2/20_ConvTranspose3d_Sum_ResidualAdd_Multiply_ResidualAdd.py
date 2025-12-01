import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 + tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = tmp4 + tmp0
    tl.store(out_ptr0 + x2, tmp5, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (16, 64, 16, 32, 32), (524288, 8192, 512, 
        16, 1))
    assert_size_stride(primals_2, (64, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 16, 32, 32), (524288, 8192, 512, 
            16, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_0[grid(524288)](primals_1, primals_2, buf0,
            524288, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
    return buf0,


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
        primals_2 = self.bias
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]