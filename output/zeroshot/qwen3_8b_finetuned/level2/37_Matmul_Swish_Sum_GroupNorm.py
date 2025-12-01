import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_mul_sigmoid_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 + tmp1
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0 + 64 * x2), xmask)
    tmp16 = tl.load(in_ptr2 + (x0 + 64 * x2), xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 0)[:, None]
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tmp8 - tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.where(xmask, tmp11, 0)
    tmp13 = tl.sum(tmp12, 0)[:, None]
    tmp14 = 64.0
    tmp15 = tmp13 / tmp14
    tmp17 = tmp16 + tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp23 = tl.where(xmask, tmp22, 0)
    tmp24 = tl.sum(tmp23, 0)[:, None]
    tmp25 = tl.load(in_ptr0 + x3, xmask)
    tmp26 = tmp25 - tmp24
    tmp27 = tmp26 * tmp20
    tmp28 = tl.load(in_ptr1 + x3, xmask)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr2 + x3, xmask)
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tl.store(out_ptr1 + x3, tmp15, xmask)
    tl.store(out_ptr2 + x3, tmp20, xmask)
    tl.store(out_ptr3 + x3, tmp31, xmask)


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
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((32768, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_0[grid(134217728)](buf0, primals_2,
            buf1, 134217728, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((524288, 64), (64, 1), torch.float32)
        buf3 = empty_strided_cuda((524288, 64), (64, 1), torch.float32)
        buf4 = empty_strided_cuda((524288, 64), (64, 1), torch.float32)
        buf5 = empty_strided_cuda((32768, 4096), (4096, 1), torch.float32)
        triton_poi_fused_native_group_norm_1[grid(2097152)](buf1,
            primals_4, primals_5, buf2, buf3, buf4, buf5, 2097152, XBLOCK=
            64, num_warps=4, num_stages=1)
        del primals_4
        del primals_5
    return buf5, primals_3, buf1, buf2, buf3, buf4, primals_3


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
        primals_2 = self.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]