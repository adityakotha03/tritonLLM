import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_native_group_norm_0(in_out_ptr0, in_ptr0, in_ptr1,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 4
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 64.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp22, xmask)
    tl.store(out_ptr0 + (r1 + 64 * x0), tmp25, xmask)


@triton.jit
def triton_poi_fused_clamp_max_convolution_1(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 307200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 12100 % 64
    x0 = xindex % 12100
    x4 = xindex // 12100
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x4, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp4, tmp7)
    tmp9 = 0.0
    tmp10 = tmp8 <= tmp9
    tl.store(out_ptr0 + x3, tmp8, xmask)
    tl.store(out_ptr1 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 30
    x1 = xindex // 30 % 30
    x2 = xindex // 900
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (60 + 2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (61 + 2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (60 + 2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (61 + 2 * x0 + 120 * x1 + 3600 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp4 = tmp3 > tmp1
    tmp5 = tmp5 > tmp3
    tmp6 = tmp4 | tmp5
    tmp9 = tmp8 > tmp7
    tmp11 = tmp10 > tmp8
    tmp12 = tmp12 > tmp10
    tmp13 = tmp11 | tmp12
    tmp14 = tmp9 | tmp13
    tmp15 = tmp6 | tmp14
    tmp16 = tmp1 > tmp0
    tmp17 = tmp3 > tmp1
    tmp18 = tmp5 > tmp3
    tmp19 = tmp17 | tmp18
    tmp20 = tmp16 | tmp19
    tmp21 = tmp8 > tmp7
    tmp22 = tmp10 > tmp8
    tmp23 = tmp12 > tmp10
    tmp24 = tmp22 | tmp23
    tmp25 = tmp21 | tmp24
    tmp26 = tmp20 | tmp25
    tmp27 = tmp26 & tmp15
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tl.store(out_ptr1 + x3, tmp27, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128, 1
        ))
    assert_size_stride(primals_4, (16, 64), (64, 1))
    assert_size_stride(primals_5, (16,), (1,))
    assert_size_stride(primals_6, (64,), (1,))
    assert_size_stride(primals_7, (64, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 126, 126), (1032192, 15504, 126, 
            1))
        buf1 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        buf3 = reinterpret_tensor(buf1, (4, 1), (1, 1), 0)
        del buf1
        buf4 = reinterpret_tensor(buf2, (4, 1), (1, 1), 0)
        del buf2
        get_raw_stream(0)
        triton_per_fused_native_group_norm_0[grid(4)](buf3, buf0, primals_4,
            buf4, 4, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_4
        buf5 = empty_strided_cuda((128, 64, 126, 126), (1032192, 15504, 126,
            1), torch.float32)
        buf6 = empty_strided_cuda((128, 64, 126, 126), (1032192, 15504, 126,
            1), torch.bool)
        triton_poi_fused_clamp_max_convolution_1[grid(307200)](buf0, buf3,
            primals_6, buf5, buf6, 307200, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf0
        del primals_6
        buf7 = empty_strided_cuda((128, 64, 30, 30), (57600, 900, 30, 1),
            torch.float32)
        buf8 = empty_strided_cuda((128, 64, 30, 30), (57600, 900, 30, 1),
            torch.bool)
        triton_poi_fused_max_pool2d_with_indices_2[grid(76800)](buf5, buf7,
            buf8, 76800, XBLOCK=512, num_warps=8, num_stages=1)
        del buf5
    return buf7, primals_1, primals_3, primals_5, buf3, buf4, buf6, buf8


class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups,
        scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_6 = self.scale
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
