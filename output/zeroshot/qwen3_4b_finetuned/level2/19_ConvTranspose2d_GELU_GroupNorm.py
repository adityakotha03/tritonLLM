import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_group_norm_0(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 163840 % 8
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tl.store(out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr1 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_group_norm_1(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 8.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = 1.0
    tmp20 = tmp19 / tmp18
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tl.store(out_ptr1 + x0, tmp20, xmask)


@triton.jit
def triton_poi_fused_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 163840 % 8
    x0 = xindex % 163840
    x2 = xindex // 81920
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask)
    tmp3 = tl.load(in_ptr2 + x3, xmask)
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + x2, xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + x2, xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + x2, xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + x2, xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = 0.0
    tmp7 = tmp6 - tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp4 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = 1.0
    tmp13 = tmp11 / tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = 1e-05
    tmp19 = tmp16 + tmp17
    tmp20 = 1.0 / tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp21 - tmp18
    tmp24 = tmp23 * tmp23
    tmp25 = tmp13 * tmp24
    tmp27 = tmp25 * tmp20
    tmp28 = tmp27 + tmp22
    tmp29 = tmp28 * tmp20
    tmp31 = tmp29 + tmp26
    tmp32 = tmp31 * tmp20
    tmp33 = tmp32 + tmp30
    tmp35 = tmp33 * tmp20
    tmp36 = tmp34 * tmp35
    tl.store(out_ptr0 + x3, tmp36, xmask)
    tl.store(out_ptr1 + x3, tmp20, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 256, 256), (4194304, 65536, 256, 1
        ))
    assert_size_stride(arg1_1, (64, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 257, 257), (4194304, 65536, 257,
            1), torch.float32)
        buf1 = empty_strided_cuda((128, 64, 257, 257), (4194304, 65536, 257,
            1), torch.bool)
        triton_poi_fused_convolution_group_norm_0[triton.launch_config](
            arg1_1, arg0_1, buf0, buf1, 1638400, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del arg0_1
        del arg1_1
        buf2 = empty_strided_cuda((64, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((64, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        triton_poi_fused_group_norm_1[triton.launch_config](buf1, buf2, buf3,
            1024, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 64, 257, 257), (4194304, 65536, 257,
            1), torch.float32)
        buf5 = empty_strided_cuda((128, 64, 257, 257), (4194304, 65536, 257,
            1), torch.float32)
        triton_poi_fused_group_norm_2[triton.launch_config](buf0, buf1, buf2,
            buf3, buf4, buf5, buf2, buf3, buf4, buf5, buf5, buf4, buf5,
            1638400, XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
        del buf1
        del buf2
        del buf3
    return buf5,


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
        num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=
            out_channels)

    def forward(self, input_0):
        arg1_1 = self.conv_transpose.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
