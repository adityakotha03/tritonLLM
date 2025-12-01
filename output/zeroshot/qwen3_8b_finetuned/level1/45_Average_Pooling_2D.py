import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = 121.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 64, 2048, 2048), (8388608, 131072, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 2038, 2038), (8388608, 131072, 
            64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(16777216)](arg0_1, buf0, 16777216,
            XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]