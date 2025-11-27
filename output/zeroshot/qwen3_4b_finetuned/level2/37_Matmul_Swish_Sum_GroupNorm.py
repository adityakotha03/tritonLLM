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
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 + tmp1
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 65536 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 65536 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 65536 * x1), xmask)
    tmp4 = tl.load(in_ptr1 + (16384 + x0 + 65536 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (32768 + x0 + 65536 * x1), xmask)
    tmp8 = tl.load(in_ptr1 + (32768 + x0 + 65536 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (49152 + x0 + 65536 * x1), xmask)
    tmp12 = tl.load(in_ptr1 + (49152 + x0 + 65536 * x1), xmask)
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
    tl.store(out_ptr0 + x2, tmp16, xmask)
    tl.store(out_ptr1 + x2, tmp28, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp8 = tmp2 * tmp6
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 + tmp5
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 1024), (1024, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (32768, 1024), (1024, 1))
    assert_size_stride(primals_4, (64, 4096), (4096, 1))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_0[grid(134217728)](primals_3,
            primals_1, buf0, 134217728, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        del primals_3
        buf1 = empty_strided_cuda((32768, 4096), (4096, 1), torch.float32)
        buf2 = empty_strided_cuda((32768, 1), (1, 1), torch.float32)
        buf3 = empty_strided_cuda((32768, 1), (1, 1), torch.float32)
        triton_poi_fused_native_group_norm_1[grid(65536)](buf0, primals_2,
            buf2, buf3, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((32768, 4096), (4096, 1), torch.float32)
        triton_poi_fused_native_group_norm_2[grid(524288)](buf0, primals_2,
            buf2, buf3, primals_4, buf4, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del buf2
        del buf3
        del primals_2
        del primals_4
    return buf4, primals_5, reinterpret_tensor(buf0, (32768, 4096), (1, 4096), 0)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
