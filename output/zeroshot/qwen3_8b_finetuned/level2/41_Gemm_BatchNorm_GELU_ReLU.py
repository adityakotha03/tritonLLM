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
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4096 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp13 = tmp4 - tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 256.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp12 * tmp23
    tl.store(out_ptr0 + (r1 + 4096 * x0), tmp1, xmask)
    tl.store(out_ptr1 + (r1 + 4096 * x0), tmp11, xmask)
    tl.store(out_ptr2 + (r1 + 4096 * x0), tmp22, xmask)
    tl.store(out_ptr3 + (r1 + 4096 * x0), tmp24, xmask)


@triton.jit
def triton_poi_fused_gelu_relu_threshold_backward_1(in_ptr0, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4096 * x0), xmask, other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tl.full([1, 1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + (r1 + 4096 * x0), tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16384, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096, 4096), (4096, 1))
    assert_size_stride(primals_3, (4096,), (1,))
    assert_size_stride(primals_4, (4096,), (1,))
    assert_size_stride(primals_5, (4096,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (4096, 
            4096), (1, 4096), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        buf2 = empty_strided_cuda((4096,), (1,), torch.float32)
        buf3 = empty_strided_cuda((4096,), (1,), torch.float32)
        buf4 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_0[grid(16384)](buf0,
            primals_3, primals_4, primals_5, buf1, buf2, buf3, buf4, 16384,
            16, XBLOCK=256, num_warps=8, num_stages=1)
        del primals_4
        del primals_5
        buf5 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_gelu_relu_threshold_backward_1[grid(1024)](buf1,
            buf5, 1024, 64, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
    return buf5, primals_1, primals_3, buf0, buf2, buf3, buf4


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, input_0):
        primals_2 = self.gemm.weight
        primals_3 = self.gemm.bias
        primals_4 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]