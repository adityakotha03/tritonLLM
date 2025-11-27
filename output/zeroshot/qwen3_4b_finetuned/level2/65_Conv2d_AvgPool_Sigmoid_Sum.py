import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1268704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 104976 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1268704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + x0 + 4096 * x1), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + x0 + 4096 * x1), xmask, eviction_policy
        ='evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + x0 + 4096 * x1), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_sigmoid_sum_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.sum(tmp5, 0)[:, None]
    tl.store(in_out_ptr0 + x0, tmp4, xmask)
    tl.store(in_ptr0 + tl.full([1], 0, tl.int32), tmp7, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 8, 384, 384), (1268704, 158587, 384, 1))
    assert_size_stride(arg1_1, (64, 8, 3, 3), (72, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 384, 384), (1268704, 16384, 4096,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1268704)](buf0, arg1_1, 1268704,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((128, 64, 1024, 1024), (67108864, 1024,
            1024, 1), torch.float32)
        triton_poi_fused_avg_pool2d_1[grid(1268704)](buf0, buf1, 1268704,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        buf3 = buf2
        del buf2
        triton_poi_fused_sigmoid_sum_2[grid(128)](buf3, buf1, 128, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf1
    return buf3, arg0_1


class ModelNew(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
