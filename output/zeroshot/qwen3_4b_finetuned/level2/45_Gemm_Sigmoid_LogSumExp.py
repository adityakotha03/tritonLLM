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
def triton_poi_fused_sigmoid_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + x2, tmp3, xmask)


@triton.jit
def triton_per_fused_logsumexp_1(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp4, xmask)


@triton.jit
def triton_per_fused_logsumexp_2(in_out_ptr0, in_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 1024 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, float('-inf'))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl_math.log(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (r1 + 1024 * x0), tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 2048), (2048, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 2048), (2048, 1))
    assert_size_stride(primals_4, (1024, 4096), (4096, 1))
    assert_size_stride(primals_5, (1024,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_sigmoid_0[grid(8388608)](buf0, primals_1, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_sigmoid_0[grid(8388608)](buf1, primals_2, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        triton_per_fused_logsumexp_1[grid(16384)](buf0, buf2, 16384, 1024,
            XBLOCK=16, num_warps=4, num_stages=1)
        buf3 = reinterpret_tensor(buf0, (16384, 1024), (1024, 1), 0)
        del buf0
        triton_per_fused_logsumexp_2[grid(16384)](buf3, primals_4, 16384,
            1024, XBLOCK=16, num_warps=4, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        triton_poi_fused_sigmoid_0[grid(8388608)](buf4, primals_5, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf5 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        triton_per_fused_logsumexp_1[grid(16384)](buf1, buf5, 16384, 1024,
            XBLOCK=16, num_warps=4, num_stages=1)
        buf6 = reinterpret_tensor(buf1, (16384, 1024), (1024, 1), 0)
        del buf1
        triton_per_fused_logsumexp_2[grid(16384)](buf6, primals_3, 16384,
            1024, XBLOCK=16, num_warps=4, num_stages=1)
        del primals_3
    return buf6, buf2, buf3, buf5


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_0):
        primals_1 = self.linear1.weight
        primals_2 = self.linear1.bias
        primals_4 = self.linear2.weight
        primals_5 = self.linear2.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
