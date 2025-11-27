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
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)
    tl.store(out_ptr2 + x0, tmp23, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 663552
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
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 4096), (4096, 1))
    assert_size_stride(primals_4, (4096,), (1,))
    assert_size_stride(primals_5, (4096,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_0[grid(4096)](primals_3,
            buf0, buf0, buf0, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_1[grid(663552)](primals_3,
            buf0, buf0, primals_1, primals_2, buf1, 663552, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf2 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_gelu_2[grid(663552)](buf1, buf2, 663552, XBLOCK=
            512, num_warps=8, num_stages=1)
        buf3 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_relu_3[grid(663552)](buf2, buf3, 663552, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_3, primals_4, primals_5, reinterpret_tensor(buf0,
        (4096, 4096), (1, 4096), 0), reinterpret_tensor(buf1, (16384, 4096),
        (4096, 1), 0)


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
        primals_4 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
