import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_hardswish_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 * tmp2
    tmp4 = 3.0
    tmp5 = tmp3 * tmp4
    tmp6 = 3.0
    tmp7 = tmp5 > tmp6
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr0 + (512000 + x0), tmp5, xmask)
    tl.store(out_ptr0 + (1024000 + x0), tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 > 0.0
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp0, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 > 0.0
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp0, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_hardswish_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 * tmp2
    tmp4 = 3.0
    tmp5 = tmp3 * tmp4
    tmp6 = 3.0
    tmp7 = tmp5 > tmp6
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr0 + (512 + x0), tmp5, xmask)
    tl.store(out_ptr0 + (1024 + x0), tmp7, xmask)


@triton.jit
def triton_poi_fused_hardswish_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 > 0.0
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp0, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_group_norm_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr1 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp0 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp3 - tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp7 - tmp16
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp11 - tmp16
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 / tmp15
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tl.store(out_ptr1 + x0, tmp28, xmask)


@triton.jit
def triton_poi_fused_group_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp6 = tl.load(in_ptr3 + x0, xmask)
    tmp9 = tl.load(in_ptr4 + x0, xmask)
    tmp12 = tl.load(in_ptr5 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = 1e-05
    tmp7 = tmp6 + tmp5
    tmp8 = tmp4 / tmp7
    tmp10 = tmp9 * tmp8
    tmp11 = tmp10 * tmp12
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 3, 16, 32, 32), (49152, 16384, 32768,
        1024, 32))
    assert_size_stride(arg1_1, (16, 3, 4, 4, 4), (192, 64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 16, 16, 32, 32), (8388608, 524288,
            32768, 1024, 32), torch.float32)
        get_ptr0 = extern_kernels.convolution
        buf1 = empty_strided_cuda((1024, 16, 16, 32, 32), (8388608, 524288,
            32768, 1024, 32), torch.float32)
        triton_poi_fused_hardswish_0[grid(524288)](arg1_1, buf1, 524288,
            XBLOCK=128, num_warps=8, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((1024, 16, 16, 32, 32), (8388608, 524288,
            32768, 1024, 32), torch.float32)
        triton_poi_fused__to_copy_1[grid(524288)](buf1, buf2, 524288,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((1024, 16, 16, 32, 32), (8388608, 524288,
            32768, 1024, 32), torch.float32)
        triton_poi_fused_hardswish_0[grid(524288)](buf2, buf3, 524288,
            XBLOCK=128, num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((1024, 16, 16, 32, 32), (8388608, 524288,
            32768, 1024, 32), torch.float32)
        triton_poi_fused__to_copy_1[grid(524288)](buf3, buf4, 524288,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        extern_kernels.convolution(arg0_1, buf4, (1024, 16, 1, 1, 1), (16384,
            1024, 1024, 1, 1), stride=(1, 0, 0, 0, 0), padding=(0, 0, 0, 0, 0
            ), dilation=(1, 1, 1, 1, 1), transposed=False, output_padding=(
            0, 0, 0, 0, 0), groups=1, bias=None)
        del arg0_1
        buf6 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        buf7 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused_group_norm_5[grid(1024)](buf5, buf6, buf7, buf7, 
            1024, XBLOCK=128, num_warps=4, num_stages=1)
        buf8 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused_group_norm_6[grid(1024)](buf5, buf6, buf7, buf7,
            buf5, buf7, buf8, 1024, XBLOCK=128, num_warps=4, num_stages=1)
        buf9 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused_hardswish_3[grid(1024)](buf8, buf9, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf10 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused__to_copy_2[grid(1024)](buf9, buf10, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf11 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused_hardswish_4[grid(1024)](buf10, buf11, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf10
    return buf11, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf11


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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
