import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_tanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 1.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp2, tmp3)
    tmp13 = 1.7154117942512922
    tmp14 = tmp12 * tmp13
    tl.store(in_out_ptr0 + x2, tmp14, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 8, 256, 256), (524288, 65536, 256, 1))
    assert_size_stride(arg1_1, (64, 8, 3, 3), (72, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 254, 254), (4124448, 64, 167, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(4096)](buf0, arg1_1, 4096, XBLOCK=256,
            num_warps=4, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 64, 254, 254), (4124448, 64, 167, 1),
            torch.float32)
        triton_poi_fused_mul_1[grid(4096)](buf0, buf1, 4096, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf0
    return buf1, arg0_1


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg1_2 = self.conv.bias
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
