import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (16 * x0 + 16 * x1 + 65536 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + 16 * x0 + 16 * x1 + 65536 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (16 + 16 * x0 + 16 * x1 + 65536 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (17 + 16 * x0 + 16 * x1 + 65536 * x2), xmask)
    tmp2 = tmp1 > tmp0
    tmp4 = tmp3 > tmp1
    tmp6 = tmp5 > tmp3
    tmp7 = tl.full([1], 0, tl.int8)
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.where(tmp6, tmp8, tmp10)
    tmp12 = tmp5 > tmp3
    tmp13 = tl.full([1], 2, tl.int8)
    tmp14 = tl.where(tmp12, tmp13, tmp11)
    tl.store(out_ptr0 + x3, tmp14, xmask)
    tl.store(out_ptr1 + x3, tmp5, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32, 64, 512, 512), (2097152, 32768, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 64, 256, 256), (16777216, 262144, 65536,
            1), torch.int8)
        buf1 = empty_strided_cuda((32, 64, 256, 256), (16777216, 262144, 65536,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(8192)](arg0_1, buf0,
            buf1, 8192, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf1, buf0


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
