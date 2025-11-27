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
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 16 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
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
    tmp31 = 0.0625
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + x0, tmp32, xmask)


@triton.jit
def triton_poi_fused_gelu_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_max_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 16 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tl.store(out_ptr0 + x0, tmp30, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (8192, 8192), (1, 8192), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(8192)](buf0, buf1, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_gelu_mul_1[grid(8192)](buf1, buf2, 8192, XBLOCK=
            256, num_warps=4, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((1024,), (1,), torch.float32)
        triton_poi_fused_max_2[grid(1024)](buf2, buf3, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_3, buf0


class ModelNew(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
