import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 103680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 3240 % 24
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8
    x3 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, r1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 / tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp6, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.where(xmask, tmp8, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([1, 1], 16, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp11 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tl.store(out_ptr2 + (x2 + 8 * x3), tmp19, xmask)
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tl.store(out_ptr1 + x0, tmp19, xmask)


@triton.jit
def triton_per_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8
    x3 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp0
    tmp4 = tmp2 - tmp3
    tmp6 = tmp5 * tmp4
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_per_fused_native_group_norm_3(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8
    x3 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp0
    tmp4 = tmp2 - tmp3
    tmp6 = tmp5 * tmp4
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp15 + tmp17
    tmp18 = libdevice.rsqrt(tmp16)
    tl.store(in_out_ptr0 + x0, tmp18, xmask)


@triton.jit
def triton_per_fused_native_group_norm_4(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8
    x3 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp0
    tmp4 = tmp2 - tmp3
    tmp6 = tmp5 * tmp4
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp15 + tmp17
    tmp18 = libdevice.rsqrt(tmp16)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1, 1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = tmp11 * tmp22
    tl.store(in_out_ptr0 + x0, tmp23, xmask)


@triton.jit
def triton_per_fused_native_group_norm_5(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8
    x3 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp0
    tmp4 = tmp2 - tmp3
    tmp6 = tmp5 * tmp4
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp15 + tmp17
    tmp18 = libdevice.rsqrt(tmp16)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1, 1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = tmp11 * tmp22
    tmp24 = tmp11 / tmp16
    tmp25 = tmp24 * tmp22
    tmp26 = tmp11 + tmp3
    tmp27 = tmp26 / tmp16
    tmp28 = tmp25 - tmp27
    tmp29 = tmp28 * tmp22
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = tmp30 / tmp20
    tmp32 = tmp29 + tmp31
    tl.store(in_out_ptr0 + x0, tmp32, xmask)


@triton.jit
def triton_per_fused_mean_6(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_per_fused_mean_7(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp5, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (24,), (1,))
    assert_size_stride(primals_3, (128, 3, 24, 32, 32), (589824, 196608, 
        8192, 256, 8))
    assert_size_stride(primals_4, (8,), (1,))
    assert_size_stride(primals_5, (8,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 24, 22, 30, 30), (475200, 196608, 
            8976, 299, 10))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(103680)](buf1, primals_2, 
            103680, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf3 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf5 = reinterpret_tensor(buf3, (128, 1), (1, 128), 0)
        del buf3
        triton_per_fused_native_group_norm_1[grid(128)](buf1, buf2, buf4,
            buf5, 128, 16, XBLOCK=1, num_warps=2, num_stages=1)
        buf6 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        triton_per_fused_native_group_norm_2[grid(128)](buf1, buf2, buf4,
            buf5, primals_4, primals_5, buf6, 128, 16, XBLOCK=32,
            num_warps=4, num_stages=1)
        del buf2
        buf7 = buf4
        del buf4
        triton_per_fused_native_group_norm_3[grid(128)](buf7, buf1, buf5,
            primals_4, primals_5, primals_5, 128, 16, XBLOCK=1, num_warps=
            2, num_stages=1)
        buf8 = buf5
        del buf5
        triton_per_fused_native_group_norm_4[grid(128)](buf8, buf1, buf7,
            primals_4, primals_5, primals_5, primals_4, 128, 16, XBLOCK=1,
            num_warps=2, num_stages=1)
        del buf1
        del primals_4
        del primals_5
        buf9 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        triton_per_fused_native_group_norm_5[grid(128)](buf9, buf1, buf7,
            primals_4, primals_5, primals_5, primals_4, 128, 16, XBLOCK=1,
            num_warps=2, num_stages=1)
        del buf7
        del primals_4
        del primals_5
        buf10 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        triton_per_fused_mean_6[grid(128)](buf9, buf10, 128, 128, XBLOCK=32,
            num_warps=4, num_stages=1)
        buf11 = buf9
        del buf9
        triton_per_fused_mean_7[grid(1)](buf11, buf10, 1, 128, XBLOCK=1,
            num_warps=2, num_stages=1)
        del buf10
    return buf11, primals_1, primals_3, buf6, buf8, buf11


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
