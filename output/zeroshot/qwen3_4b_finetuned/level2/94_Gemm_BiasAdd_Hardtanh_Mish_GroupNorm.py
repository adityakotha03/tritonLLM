import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_hardtanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_hardtanh_mish_1(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp6 = 1.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp8 * tmp3
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + x0, tmp12, xmask)


@triton.jit
def triton_per_fused__native_group_norm_add_clamp_2(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex % 2
    r2 = rindex // 2
    x0 = xindex % 4
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp3 = tl.load(in_ptr0 + (4 + r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp4 = tl.load(in_ptr1 + (4 + r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp7 = tl.load(in_ptr0 + (1 + r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp8 = tl.load(in_ptr1 + (1 + r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp11 = tl.load(in_ptr0 + (5 + r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp12 = tl.load(in_ptr1 + (5 + r1 + 2 * x0 + 8 * r2 + 1024 * x1), xmask,
        other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp2 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp5 - tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 - tmp16
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp13 - tmp16
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 / tmp15
    tl.store(out_ptr0 + x3, tmp16, xmask)
    tl.store(out_ptr1 + x3, tmp28, xmask)
    tl.store(out_ptr2 + x3, tmp15, xmask)


@triton.jit
def triton_poi_fused__native_group_norm_add_clamp_3(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = tmp7 * tmp10
    tmp9 = tmp8 + tmp12
    tmp11 = tmp9 * tmp14
    tl.store(out_ptr0 + x2, tmp11, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (8192,), (1,))
    assert_size_stride(primals_5, (4,), (1,))
    assert_size_stride(primals_6, (4,), (1,))
    assert_size_stride(primals_7, (4,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_0[grid(8388608)](buf0, primals_2,
            buf1, 8388608, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_hardtanh_mish_1[grid(8388608)](buf1, primals_4,
            primals_5, buf2, 8388608, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_4
        del primals_5
        buf3 = empty_strided_cuda((1024, 4), (4, 1), torch.float32)
        buf4 = empty_strided_cuda((1024, 4), (4, 1), torch.float32)
        buf5 = empty_strided_cuda((1024, 4), (4, 1), torch.float32)
        triton_per_fused__native_group_norm_add_clamp_2[grid(16)](buf2,
            primals_6, buf3, buf4, buf5, 16, 128, XBLOCK=1, num_warps=2,
            num_stages=1)
        buf6 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused__native_group_norm_add_clamp_3[grid(8388608)](buf2,
            primals_6, buf3, buf4, buf5, primals_7, buf6, 8388608, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del buf3
        del buf4
        del buf5
        del primals_7
    return buf6, primals_3, primals_6, buf0, buf1, buf2


class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_4 = self.bias
        primals_5 = self.groupnorm.weight
        primals_6 = self.groupnorm.bias
        primals_7 = self.groupnorm.num_groups
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
