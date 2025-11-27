import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = xindex // 3 % 40
    x2 = xindex // 120 % 40
    x3 = xindex // 4800
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x2 + 5120 * x1 + 204800 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (3 + x0 + 128 * x2 + 5120 * x1 + 204800 * x3),
        xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + x0 + 128 * x2 + 5120 * x1 + 204800 * x3),
        xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (67 + x0 + 128 * x2 + 5120 * x1 + 204800 * x3),
        xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (128 + x0 + 128 * x2 + 5120 * x1 + 204800 * x3
        ), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (131 + x0 + 128 * x2 + 5120 * x1 + 204800 * x3
        ), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x0 + 128 * x2 + 5120 * x1 + 204800 *
        x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (259 + x0 + 128 * x2 + 5120 * x1 + 204800 *
        x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (384 + x0 + 128 * x2 + 5120 * x1 + 204800 *
        x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (387 + x0 + 128 * x2 + 5120 * x1 + 204800 *
        x3), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + x4, tmp18, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 128, 128, 128), (524288, 16384, 128,
        128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 40, 40, 3), (19200, 600, 15, 3.75,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool3d_with_indices_0[grid(13824)](arg0_1, buf0,
            13824, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
