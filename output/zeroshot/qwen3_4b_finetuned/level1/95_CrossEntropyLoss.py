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


@triton.jit
def triton_per_fused__log_softmax_0(in_ptr0, out_ptr2, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4096 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp5 - tmp11
    tl.store(out_ptr2 + (r1 + 4096 * x0), tmp12, xmask)


@triton.jit
def triton_per_fused__log_softmax_div_mul_neg_sum_1(in_out_ptr0, in_ptr0,
    in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_out_ptr0 + r0, None)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(0 <= r0, tmp2, float('-inf'))
    tmp5 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(0 <= r0, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = -tmp13
    tmp15 = 0.015625
    tmp16 = tmp14 * tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp16, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 4096), (4096, 1))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((32768, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__log_softmax_0[grid(32768)](arg0_1, buf2, 32768, 
            4096, XBLOCK=32, num_warps=4, num_stages=1)
        del arg0_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3
        del buf3
        triton_per_fused__log_softmax_div_mul_neg_sum_1[grid(1)](buf4, arg1_1,
            buf2, 1, 4096, XBLOCK=1, num_warps=2, num_stages=1)
        del arg1_1
        del buf2
    return buf4,


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
