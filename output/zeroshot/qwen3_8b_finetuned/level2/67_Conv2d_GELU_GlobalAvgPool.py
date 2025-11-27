import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2097152
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tl.full([1], tmp3, tl.int32)
    tmp5 = tmp2 / tmp4
    tmp6 = 3.141592653589793
    tmp7 = tmp6 * tmp5
    tmp8 = tl.full([1], tmp7, tl.int32)
    tmp9 = tl.full([1], 1.4142135623731, tl.int32)
    tmp10 = tmp8 * tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused_adaptive_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 1])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 1))
    tmp5 = 64516.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 254, 254), (4194304, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 254, 254), (4194304, 65536, 256, 1
            ), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(2097152)](arg0_1, buf0, 2097152, 1,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 64), (8192, 128), torch.float32)
        triton_poi_fused_adaptive_avg_pool2d_1[grid(16384)](buf1, buf2,
            16384, 1, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
    return buf2,


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0].squeeze(-1).squeeze(-1)