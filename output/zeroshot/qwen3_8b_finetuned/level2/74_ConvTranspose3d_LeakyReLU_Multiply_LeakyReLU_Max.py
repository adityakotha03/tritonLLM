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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
empty_cuda = torch._C._dynamo.guards._empty_cuda
reinterpret_tensor_1 = torch._C._dynamo.guards._reinterpret_tensor_1
empty_strided_cuda_1 = torch._C._dynamo.guards._empty_strided_cuda_1


@triton.jit
def triton_leaky_relu1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tl_math.maximum(tmp0, tmp1 * 0.2)
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_leaky_relu2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tl_math.maximum(tmp0, tmp1 * 0.2)
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_mul(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool_0(in_ptr0, out_ptr0, xnumel, xoffset, rnumel,
    roffset, XBLOCK: tl.constexpr):
    xoffset = 3
    xnumel = 8192
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (x2 + 1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x2 + 2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x2 + 3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x2 + 4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (x2 + 5), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x2 + 6), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (x2 + 7), xmask, eviction_policy='evict_last')
    tmp2 = tl_math.maximum(tmp0, tmp1)
    tmp4 = tl_math.maximum(tmp2, tmp3)
    tmp5 = tl_math.maximum(tmp4, tmp6)
    tmp7 = tl_math.maximum(tmp5, tmp9)
    tmp8 = tl_math.maximum(tmp7, tmp12)
    tmp10 = tl_math.maximum(tmp8, tmp15)
    tmp11 = tl_math.maximum(tmp10, tmp18)
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_2, (16, 16, 16, 32, 32), (32768, 2048, 128, 64,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 16, 32, 32), (32768, 2048, 128, 64,
            1), torch.float32)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        buf5 = buf4
        del buf4
        get_raw_stream(0)
        triton_leaky_relu1[grid(16384)](buf5, primals_2, buf5, 16384, 1,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf6 = empty_strided_cuda((16, 32, 16, 32, 32), (32768, 2048, 128, 64,
            1), torch.float32)
        buf7 = buf6
        del buf6
        buf8 = buf7
        del buf7
        triton_mul[grid(16384)](buf8, primals_1, buf8, 16384, 1, XBLOCK=128,
            num_warps=4, num_stages=1)
        del primals_1
        buf9 = buf8
        del buf8
        buf10 = buf9
        del buf9
        triton_leaky_relu2[grid(16384)](buf10, buf6, buf10, 16384, 1,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf6
        buf11 = empty_strided_cuda((16, 32, 8, 16, 16), (4096, 128, 8, 64, 4),
            torch.float32)
        buf12 = buf11
        del buf11
        triton_poi_fused_max_pool_0[grid(8192)](buf10, buf12, 8192, 3, 1,
            0, XBLOCK=128, num_warps=4, num_stages=1)
        del buf10
        del buf12
    return buf11, reinterpret_tensor_1(buf9, (16, 32, 16, 32, 32), (32768,
        2048, 128, 64, 1), 0), reinterpret_tensor(buf6, (32, 1, 1, 1), (1,
        1, 1, 1), 0)


class ModelNew(nn.Module):
    """
    Optimized model with Triton kernels for LeakyReLU, elementwise multiplication, and max pooling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
        output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_0):
        primals_2 = input_0
        primals_1 = self.multiplier
        output = call([primals_1, primals_2])
        return output[0]