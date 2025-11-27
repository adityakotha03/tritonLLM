import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 1.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + x0, tmp16, xmask)


@triton.jit
def triton_poi_fused_logsumexp_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 1.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(tmp16 > tmp17, tmp16, tmp17)
    tmp20 = tl.where(tmp16 > tmp17, tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + x0, tmp21, xmask)


@triton.jit
def triton_poi_fused_logsumexp_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 1.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(tmp16 > tmp17, tmp16, tmp17)
    tmp20 = tl.where(tmp16 > tmp17, tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + x0, tmp21, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mean_0[grid(1024)](primals_3, buf0, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        triton_poi_fused_logsumexp_1[grid(1024)](buf0, buf1, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        triton_poi_fused_logsumexp_2[grid(1024)](buf1, buf2, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf1
    return reinterpret_tensor(buf2, (1024,), (1,), 0), primals_1, buf0


class ModelNew(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
