import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mean_max_log_sum_exp_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8192 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = 8192.0
    tmp12 = tmp10 / tmp11
    tmp13 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = triton_helpers.maximum(tmp16, 0)
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tl_math.log(tmp22)
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tl_math.log(tmp28)
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tl_math.log(tmp34)
    tl.store(out_ptr0 + x0, tmp35, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf1 = torch.ops.aten.mm.default(arg0_1, arg1_1, out=buf0)
        del arg1_1
        buf2 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mean_max_log_sum_exp_0[grid(1024)](buf1, arg0_1,
            buf2, 1024, 8192, XBLOCK=1, num_warps=4, num_stages=1)
        del arg0_1
    return buf2,


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
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        arg1_1 = self.linear.bias
        arg0_1 = input_0
        del input_0
        arg1_1 = arg1_1.to(torch.float32)
        buf1 = self.linear.weight
        del self.linear.weight
        buf0 = torch.ops.aten.mm.default(arg0_1, buf1, out=buf1)
        del buf1
        buf2 = call([arg0_1, arg1_1])
        return buf2[0]