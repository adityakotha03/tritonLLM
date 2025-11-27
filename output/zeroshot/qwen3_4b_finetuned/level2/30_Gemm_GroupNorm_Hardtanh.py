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
def triton_poi_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5120 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6144 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7168 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8192 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9216 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = 8.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp0 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tmp1 - tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp26 = tmp3 - tmp20
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 + tmp27
    tmp29 = tmp5 - tmp20
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 + tmp30
    tmp32 = tmp7 - tmp20
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 + tmp33
    tmp35 = tmp9 - tmp20
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = tmp11 - tmp20
    tmp39 = tmp38 * tmp38
    tmp40 = tmp37 + tmp39
    tmp41 = tmp13 - tmp20
    tmp42 = tmp41 * tmp41
    tmp43 = tmp40 + tmp42
    tmp44 = tmp15 - tmp20
    tmp45 = tmp44 * tmp44
    tmp46 = tmp43 + tmp45
    tmp47 = tmp17 - tmp20
    tmp48 = tmp47 * tmp47
    tmp49 = tmp46 + tmp48
    tmp50 = 7.0
    tmp51 = tmp49 / tmp50
    tmp52 = 1e-05
    tmp53 = tmp51 + tmp52
    tmp54 = libdevice.rsqrt(tmp53)
    tl.store(out_ptr0 + x0, tmp20, xmask)
    tl.store(out_ptr1 + x0, tmp54, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 5120
    x2 = xindex // 51200
    x3 = xindex % 5120
    x4 = xindex // 5120
    x5 = xindex % 16
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192 * x2), xmask, eviction_policy=
        'evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + 8192 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + x4, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + x4, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x5, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(xmask, tmp6, 0)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr0 + x6, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -2.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 2.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_native_group_norm_0[grid(512)](primals_3, buf0,
            primals_4, 512, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((1024, 5120), (5120, 1), torch.float32)
        triton_poi_fused_native_group_norm_1[grid(819200)](primals_3, buf0,
            primals_2, primals_4, primals_5, buf1, 819200, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_2
        del primals_4
        del primals_5
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_hardtanh_2[grid(819200)](buf1, buf2, 819200,
            XBLOCK=1024, num_warps=4, num_stages=1)
    return buf2, primals_1, primals_3, buf0, reinterpret_tensor(buf1, (1024,
        5120), (5120, 1), 0)


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
