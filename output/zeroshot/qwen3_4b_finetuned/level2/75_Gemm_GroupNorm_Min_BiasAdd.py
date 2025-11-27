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
def triton_poi_fused_add_native_group_norm_0(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8192
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp7 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 1024.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(out_ptr0 + x2, tmp5, xmask)
    tl.store(out_ptr1 + x2, tmp22, xmask)
    tl.store(out_ptr2 + x2, tmp12, xmask)


@triton.jit
def triton_poi_fused_add_min_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8192
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.load(in_ptr0 + (8192 + x0 + 8192 * x1), xmask)
    tmp6 = tl.load(in_ptr1 + (8192 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (1024 + x2), xmask)
    tmp9 = tmp5 + tmp6
    tmp10 = tmp9 + tmp8
    tmp11 = triton_helpers.minimum(tmp4, tmp10)
    tl.store(out_ptr0 + x2, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (1024,), (1,))
    assert_size_stride(primals_5, (1, 8192, 1, 1), (8192, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_native_group_norm_0[grid(1024)](primals_3,
            primals_1, primals_2, buf0, buf1, buf2, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
        del primals_4
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_min_1[grid(1024)](buf0, primals_5, buf2, buf3,
            1024, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del buf2
        del primals_5
    return buf3, primals_5, buf1


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = self.group_norm.weight
        primals_4 = self.group_norm.bias
        primals_5 = self.bias
        primals_6 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6])
        return output[0]
