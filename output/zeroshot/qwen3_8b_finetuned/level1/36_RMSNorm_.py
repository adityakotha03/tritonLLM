import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_pow_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 64])
    tmp2 = tl.sum(tmp1, 1)[:, None]
    tmp3 = 64.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp4 + 1e-05
    tmp6 = tl_math.sqrt(tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_div_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (112, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(primals_2, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_pow_0[grid(18874368)](primals_1, buf1, 18874368, XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = buf1
        del buf1
        triton_poi_fused_mean_1[grid(18874368)](buf2, buf2, 18874368, XBLOCK=64, num_warps=4, num_stages=1)
        del buf2
        buf3 = empty_strided_cuda((112, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32)
        triton_poi_fused_div_2[grid(18874368)](primals_1, buf3, buf3, 18874368, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
    return buf3, primals_2


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, input_0):
        primals_1 = input_0
        primals_2 = torch.tensor(64, dtype=torch.int64, device='cuda')
        output = call([primals_1, primals_2])
        return output[0]