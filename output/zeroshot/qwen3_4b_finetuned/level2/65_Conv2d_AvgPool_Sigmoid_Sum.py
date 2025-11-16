import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = yindex // 64
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y3 + 231616 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + y0, tl.full([1, 1], y0, tl.int32), eviction_
        policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 384 * y4), tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 8, 384, 384), (12171984, 1521498,
        384, 1))
    assert_size_stride(primals_2, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 384, 384), (1521498, 1, 384, 
            384), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(32768)](primals_1, buf0, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((128, 64, 384, 384), (1521498, 1, 384, 
            384), torch.float32)
        extern_kernels.convolution(buf0, primals_2, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf2 = reinterpret_tensor(buf0, (128, 64, 383, 383), (12171984, 1,
            384, 384), 0)
        del buf0
        triton_poi_fused_convolution_1[grid(64, 384)](buf1, primals_3, buf2,
            64, 384, XBLOCK=32, YBLOCK=64, num_warps=4, num_stages=1)
        del buf1
        del primals_3
    return reinterpret_tensor(buf2, (128,), (1,), 0), primals_2


class ModelNew(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, input_0):
        primals_2 = self.conv.weight
        primals_3 = self.conv.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
