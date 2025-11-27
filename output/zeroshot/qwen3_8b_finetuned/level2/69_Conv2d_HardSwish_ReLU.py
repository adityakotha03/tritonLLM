import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_hardswish_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl_math.add(tmp1, 3)
    tmp4 = tl_math.minimum(tmp3, 6)
    tmp5 = tl_math.maximum(tmp4, 0)
    tmp6 = tl_math.multiply(tmp0, tmp5)
    tmp7 = tl_math.divide(tmp6, 6)
    tmp8 = tl_math.maximum(tmp7, 0)
    tl.store(out_ptr0 + x0, tmp8, xmask)


def triton_poi_fused_hardswish_relu_0_cuda(in_out_ptr0, input_0, input_1):
    arg0_1, arg1_1 = input_0, input_1
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(arg1_1, (128, 64, 128, 128), (1048576, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardswish_relu_0[grid(1048576)](arg1_1, buf0, 1048576
            , XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        del input_0
        del input_1
    return buf0, arg0_1


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, followed by a custom Triton
    kernel that implements the combined hardswish + ReLU operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = self.conv(arg0_1)
        arg1_1 = arg1_1
        buf0, arg0_1 = triton_poi_fused_hardswish_relu_0_cuda(arg0_1, arg1_1,
            arg0_1)
        return buf0