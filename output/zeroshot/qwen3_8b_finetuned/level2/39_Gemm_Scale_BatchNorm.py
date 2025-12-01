import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (3 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tmp0 + tmp1
    tmp6 = tmp4 - tmp2
    tmp8 = tmp6 * tmp6
    tmp10 = tmp7 + tmp1
    tmp12 = tmp10 - tmp2
    tmp14 = tmp12 * tmp12
    tmp15 = tmp8 + tmp14
    tmp16 = tmp11 + tmp1
    tmp17 = tmp16 - tmp2
    tmp18 = tmp17 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp13 + tmp1
    tmp21 = tmp20 - tmp2
    tmp22 = tmp21 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = 4095.0
    tmp25 = tmp23 / tmp24
    tmp26 = tmp3 - tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 * tmp26
    tmp29 = tmp28 / tmp24
    tmp30 = tmp29 + tmp27
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp33 = tmp32 * tmp5
    tmp34 = tmp33 + tmp4
    tl.store(out_ptr0 + x3, tmp1, xmask)
    tl.store(out_ptr1 + x3, tmp3, xmask)
    tl.store(out_ptr2 + x3, tmp5, xmask)
    tl.store(out_ptr3 + x3, tmp31, xmask)
    tl.store(out_ptr4 + x3, tmp34, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16384, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (4096,), (1,))
    assert_size_stride(primals_4, (4096,), (1,))
    assert_size_stride(primals_5, (4096,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(primals_1, primals_2, out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(67108864)](buf0, primals_3, buf1, 
            67108864, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((4096, 16384), (16384, 1), torch.float32)
        buf3 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf4 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        triton_poi_fused__native_batch_norm_legit_1[grid(67108864)](buf1,
            primals_4, primals_5, buf2, buf3, buf4, buf0, buf3, buf4, buf2,
            67108864, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        del primals_4
        del primals_5
    return buf2, primals_1, primals_3, buf0, buf2, buf3, buf4


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0):
        primals_2 = self.gemm.weight
        primals_3 = self.gemm.bias
        primals_4 = self.scale
        primals_5 = self.bn.weight
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]