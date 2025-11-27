import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9
    x1 = xindex // 9 % 128
    x2 = xindex // 1152 % 128
    x3 = xindex // 147456
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (x2 //
        4) + 65536 * x3 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (
        x2 // 4) + 65536 * x3 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (
        x2 // 4) + 65536 * x3 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (x2 // 
        4) + 65536 * x3 + 1 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + 4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (
        x2 // 4) + 65536 * x3 + 1 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), 
        xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (
        x2 // 4) + 65536 * x3 + 1 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), 
        xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (x2 // 
        4) + 65536 * x3 + 2 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), xmask,
        eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (
        x2 // 4) + 65536 * x3 + 2 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), 
        xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + 4 * (x0 // 3) + 16 * (x1 // 4) + 512 * (
        x2 // 4) + 65536 * x3 + 2 + x0 % 3 + 12 * (x1 % 4) + 64 * (x2 % 4)), 
        xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.0625
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + x4, tmp18, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 128, 128, 256), (1310720, 40960, 320,
        256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 126, 126, 254), (1306816, 40801,
            323, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool3d_0[grid(36864)](arg0_1, buf0, 36864,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
