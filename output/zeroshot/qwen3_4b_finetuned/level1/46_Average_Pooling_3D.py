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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask)
    tl.store(out_ptr0 + x3, tmp0, xmask)


def call(args):
    primals_1, = args
    args.clear()
    assert_size_stride(primals_1, (16, 32, 128, 128, 256), (1310720, 41943, 
        1024, 8, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 128, 128, 256), (1310720, 41943,
            1024, 8, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(256)](primals_1, buf0, 256,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
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
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, 
            padding=padding)

    def forward(self, input_0):
        primals_1 = self.avg_pool.weight
        primals_1 = reinterpret_tensor(primals_1, (16, 32, 3, 3, 3), (0, 1, 
            1024, 348, 333), 0)
        output = call([primals_1, input_0])
        return output[0]
