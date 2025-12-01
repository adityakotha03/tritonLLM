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
def triton_per_fused_add_div_log_mul_rsub_sum_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None)
    tmp2 = tl_math.log(tmp0)
    tmp3 = tl_math.log(tmp1)
    tmp4 = tmp2 - tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 16384.0
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + tl.full([1], 0, tl.int32), tmp10, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 2), (2, 1))
    assert_size_stride(arg1_1, (8192, 2), (2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_add_div_log_mul_rsub_sum_0[grid(1)](arg0_1, arg1_1,
            buf1, 1, 128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf1,


class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

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