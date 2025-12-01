import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_sigmoid_tanh_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.tanh(tmp2)
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.sigmoid(tmp5)
    tl.store(out_ptr0 + x2, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16, 16, 3, 3, 3), (864, 54, 18, 6, 1))
    assert_size_stride(primals_2, (128, 3, 16, 16, 16), (16384, 5461, 341,
        21, 1))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_2, primals_1,
            stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
            transposed=False, output_padding=(0, 0, 0), groups=1,
            bias=None)
        assert_size_stride(buf0, (128, 16, 16, 64, 64), (1048576, 65536, 4096,
            64, 1))
        buf1 = empty_strided_cuda((128, 16, 16, 64, 64), (1048576, 65536,
            4096, 64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_tanh_0[grid(2097152)](buf0, primals_3,
            primals_4, buf1, 2097152, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_3
        del primals_4
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.scaling_factor
        primals_4 = self.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]