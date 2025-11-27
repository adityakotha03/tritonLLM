import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 168960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 128
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = torch.minimum(tmp0, tmp1)
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(arg1_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 128, 128), (2097152, 16384, 
            128, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(168960)](arg1_1, buf0, 
            168960, XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = torch.ops.aten.convolution.default(buf0, arg0_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (128, 128, 128, 128), (2097152, 16384, 128,
            1))
        buf2 = buf1
        del buf1
        triton_poi_fused_add_mul_1[grid(2097152)](buf2, arg0_1, 2097152,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf2,


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value,
        bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
