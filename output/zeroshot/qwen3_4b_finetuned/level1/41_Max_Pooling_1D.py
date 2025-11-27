import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = xindex // 3
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8 * x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0 + 128 * x1), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0 + 128 * x1), xmask, eviction_policy
        ='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0 + 128 * x1), xmask, eviction_policy
        ='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0 + 128 * x1), xmask, eviction_policy
        ='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0 + 128 * x1), xmask, eviction_policy
        ='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0 + 128 * x1), xmask,
        eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0 + 128 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tl.store(out_ptr0 + x2, tmp14, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (64, 192, 65536), (12800064, 65536, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 192, 3), (576, 3, 192), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(3072)](arg0_1, buf0,
            3072, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, return_indices=return_indices)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
