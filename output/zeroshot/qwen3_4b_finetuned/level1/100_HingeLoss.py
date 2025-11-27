import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_clamp_mean_mul_sub_0(in_out_ptr0, in_ptr0, in_ptr1,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp3 = tl.load(in_ptr1 + r0, None)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp10 = 32768.0
    tmp11 = tmp9 / tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp11, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_clamp_mean_mul_sub_0[grid(1)](buf1, arg0_1, arg1_1,
            1, 32768, XBLOCK=1, num_warps=8, num_stages=1)
        del arg0_1
        del arg1_1
    return buf1,


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

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
