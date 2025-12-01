import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_abs_mul_rsub_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 * tmp2
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 == 0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 - tmp4
    tmp12 = tmp11 * tmp10
    tmp13 = tmp8 + tmp12
    tl.store(out_ptr0 + x0, tmp13, xmask)


@triton.jit
def triton_per_fused_add_div_sum_1(in_out_ptr0, in_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 32768.0
    tmp7 = tmp5 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp7, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), 1)
    assert_size_stride(arg1_1, (32768,), 1)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_abs_mul_rsub_0[grid(32768)](arg0_1, arg1_1,
            buf0, 32768, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1
        del buf1
        triton_per_fused_add_div_sum_1[grid(1)](buf2, buf0, 1, 1, XBLOCK=1,
            num_warps=4, num_stages=1)
        del buf0
    return buf2,


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

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