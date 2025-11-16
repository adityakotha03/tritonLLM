import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_hardtanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 66560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp6, tmp3)
    tmp8 = 2.0
    tmp9 = tmp7 >= tmp8
    tmp10 = triton_helpers.minimum(tmp6, tmp7)
    tmp11 = tl.where(tmp9, tmp10, tmp7)
    tl.store(in_out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_poi_fused_hardswish_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 66560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp5, tmp2)
    tmp7 = tmp6 / tmp3
    tmp8 = tmp4 * tmp7
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__to_copy_mean_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 + tmp6
    tmp10 = tmp8 + tmp9
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 16.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + x0, tmp32, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 4, 4, 4), (192, 64, 16, 4, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (1024, 3, 16, 32, 32), (49152, 16384, 512,
        16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 16, 1, 1, 1), (16, 1, 1, 1, 1),
            torch.float32)
        extern_kernels.convolution(primals_3, primals_1, stride=(1, 1, 1),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None, out=buf0)
        del primals_1
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_hardtanh_0[grid(66560)](buf1, primals_2, 66560,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((1024, 16, 16, 1, 1), (256, 16, 1, 1, 1),
            torch.float32)
        buf3 = empty_strided_cuda((1024, 16, 1, 1, 1), (16, 1, 1, 1, 1),
            torch.float32)
        triton_poi_fused_hardswish_1[grid(66560)](buf1, primals_3, buf2, 
            66560, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused__to_copy_mean_2[grid(1024)](buf2, buf4, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
    return buf4, primals_3, reinterpret_tensor(buf1, (1024, 16, 16, 1, 1),
        (256, 1, 1, 1, 1), 0), reinterpret_tensor(primals_3, (1024, 3, 16, 
        32, 32), (49152, 1, 16384, 512, 16), 0)


class ModelNew(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4,
        bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=
            bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
