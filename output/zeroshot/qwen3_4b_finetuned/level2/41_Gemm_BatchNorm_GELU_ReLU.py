import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4096
    y1 = yindex // 4096
    tmp0 = tl.load(in_ptr0 + (y0 + 4096 * x2 + 16777216 * y1), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + (4096 + y0 + 4096 * x2 + 16777216 * y1), xmask &
        ymask)
    tmp3 = tl.load(in_ptr0 + (8192 + y0 + 4096 * x2 + 16777216 * y1), xmask &
        ymask)
    tmp5 = tl.load(in_ptr0 + (12288 + y0 + 4096 * x2 + 16777216 * y1), xmask
        & ymask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2 + 4096 * y3), tmp8, xmask & ymask)


@triton.jit
def triton_per_fused_add_native_batch_norm_mean_1(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex
    x2 = xindex // 4096
    x0 = xindex % 4096
    x4 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x1, xmask)
    tmp2 = tl.load(in_ptr1 + x1, xmask)
    tmp5 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x4, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + x4, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + x4, xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 * tmp1
    tmp8 = tmp6 + tmp5
    tmp9 = tmp8 * tmp1
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp1
    tmp15 = tmp12 * tmp1
    tmp16 = tmp13 * tmp15
    tmp17 = tmp14 * tmp1
    tmp18 = tmp16 + tmp17
    tmp19 = 16384.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp14 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tmp12 - tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp26 = tmp11 / tmp19
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 - tmp27
    tmp29 = 15976224.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = 1.0 / tmp32
    tmp34 = tmp18 * tmp33
    tl.store(out_ptr0 + x1, tmp4, xmask)
    tl.store(out_ptr1 + x1, tmp18, xmask)
    tl.store(out_ptr2 + x1, tmp34, xmask)


@triton.jit
def triton_per_fused_relu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__to_copy_0[grid(4096, 4096)](primals_1, buf0, 4096,
            4096, XBLOCK=128, YBLOCK=256, num_warps=8, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        buf3 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        buf4 = buf3
        triton_per_fused_add_native_batch_norm_mean_1[grid(4096)](buf0,
            primals_3, primals_2, buf1, buf2, buf3, buf4, buf5, buf6, 4096,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
        del primals_2
        buf7 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_per_fused_relu_2[grid(16777216)](buf6, buf7, 16777216,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf6
    return buf7, primals_3, buf1, buf2, buf4, buf5


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
