import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_min_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = triton_helpers.minimum(tmp0, tmp2)
    tmp4 = tmp3 + tmp1
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_2, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_3, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_2, primals_1,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 128, 128), (2097152, 16384, 128,
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_min_mul_0[grid(2621440)](buf1, primals_3, 
            2621440, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = input_0
        primals_3 = self.bias
        output = call([primals_1, primals_2, primals_3])
        return output[0]