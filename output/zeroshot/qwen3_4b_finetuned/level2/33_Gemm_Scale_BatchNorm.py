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
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8398080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_per_fused_add_native_batch_norm_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8192
    x3 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (r1 + 1024 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 1024.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr2 + (r1 + 1024 * x3), tmp23, xmask)
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tl.store(out_ptr1 + x0, tmp18, xmask)


@triton.jit
def triton_poi_fused_add_native_batch_norm_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8398080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp11 = libdevice.rsqrt(tmp9)
    tmp12 = tmp4 * tmp11
    tmp13 = tmp12 * tmp10
    tl.store(out_ptr0 + x2, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (1024,), (1,))
    assert_size_stride(primals_5, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(8398080)](primals_3, primals_5, buf0,
            8398080, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf4 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        buf5 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        triton_per_fused_add_native_batch_norm_1[grid(8192)](primals_3,
            primals_2, buf1, buf2, buf4, 8192, 1024, XBLOCK=32, num_warps=4,
            num_stages=1)
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_native_batch_norm_2[grid(8398080)](primals_3,
            primals_2, buf1, buf2, buf4, buf3, 8398080, XBLOCK=512,
            num_warps=8, num_stages=1)
        del buf1
        del buf2
        del buf4
    return buf3, primals_2, primals_3, primals_4, reinterpret_tensor(buf0,
        (8192, 1024), (1, 8192), 0)


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_5 = self.scale
        primals_3 = self.bn.weight
        primals_4 = self.bn.bias
        primals_4_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_4_1])
        return output[0]
