import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x1 = xindex // 128 % 128
    x2 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1 + 8192 * x2 + 1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0 + 256 * x1 + 8192 * x2 + 2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + 256 * x1 + 8192 * x2 + 4), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (127 + x0 + 256 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (128 + x0 + 256 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (129 + x0 + 256 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (255 + x0 + 256 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (256 + x0 + 256 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (257 + x0 + 256 * x1 + 8192 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp11 = 0.125
    tmp13 = tmp12 + tmp10
    tmp15 = tmp14 + tmp13
    tmp17 = tmp16 + tmp15
    tmp18 = tmp11 * tmp17
    tl.store(out_ptr0 + x3, tmp18, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 128, 128, 256), (131072, 4096, 32, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 64, 64, 128), (131072, 4096, 64,
            1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool3d_0[grid(262144)](arg0_1, buf0, 262144,
            XBLOCK=128, num_warps=4, num_stages=1)
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