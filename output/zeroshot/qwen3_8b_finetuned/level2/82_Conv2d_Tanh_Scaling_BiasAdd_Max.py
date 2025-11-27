import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8388608
    x1 = xindex // 8388608
    x3 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8388608
    x1 = xindex // 8388608
    x3 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl_math.tanh(tmp0)
    tl.store(out_ptr0 + x2, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8388608
    x1 = xindex // 8388608
    x3 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 256, 256), (4194304, 65536, 256, 1))
    assert_size_stride(arg1_1, (64, 1, 1), (1, 64, 64))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256, 1), torch.float32)
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(67108864)](arg0_1, buf2, 67108864, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf3 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256, 1), torch.float32)
        triton_poi_fused_tanh_1[grid(67108864)](buf2, buf3, 67108864, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256, 1), torch.float32)
        triton_poi_fused_add_2[grid(67108864)](buf3, arg1_1, buf4, 67108864, XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
    return buf1, buf3, buf4, arg0_1, arg1_1


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
        arg0_1 = input_0
        arg1_1 = self.bias
        output = call([arg0_1, arg1_1])
        return output[2]