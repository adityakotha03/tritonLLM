import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_hardtanh_hardswish_mean_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 6.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (48 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (64 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (80 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (96 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (112 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (128 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (144 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (160 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = 32.0
    tmp28 = tmp26 / tmp27
    tmp29 = tmp5 - tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)
    tl.store(out_ptr1 + x2, tmp28, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 16, 16, 32, 32), (1638400, 10240, 640,
        20, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 16, 16, 32, 32), (1638400, 10240,
            640, 20, 1), torch.float32)
        buf1 = empty_strided_cuda((1024, 16, 16, 32, 32), (1638400, 10240,
            640, 20, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_hardswish_mean_0[grid(16384000)](arg0_1,
            buf0, buf1, 16384000, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf1,


class ModelNew(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, input_0):
        arg0_1 = self.conv(input_0)
        arg0_2 = arg0_1
        output = call([arg0_2])
        return output[0]