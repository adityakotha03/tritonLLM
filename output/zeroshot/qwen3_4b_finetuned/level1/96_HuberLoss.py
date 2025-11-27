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


@triton.jit
def triton_per_fused_abs_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([1], 0, tl.int32), tmp5, None)


@triton.jit
def triton_per_fused_abs_sub_sum_1(in_ptr0, in_ptr1, out_ptr1, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 256 * x0), xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr1 + tl.broadcast_to(x0, [XBLOCK, RBLOCK]), tmp6, xmask)


@triton.jit
def triton_per_fused_abs_sub_sum_2(in_ptr0, in_ptr1, out_ptr1, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 256 * x0), xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.full([1, 1], 1, tl.int32)
    tmp6 = tmp5 < tmp4
    tmp7 = tl.where(tmp6, tmp4, tmp2)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tl.store(out_ptr1 + tl.broadcast_to(x0, [XBLOCK, RBLOCK]), tmp11, xmask)


@triton.jit
def triton_per_fused_abs_sub_sum_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(triton_helpers.not_first_last([XBLOCK, RBLOCK]), tl
        .load(tmp3, other=0), tmp3)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp10, None)


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
        triton_per_fused_abs_sub_0[grid(1)](buf1, arg0_1, arg1_1, 1, 256,
            num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((32768,), (1,), torch.float32)
        triton_per_fused_abs_sub_sum_1[grid(32768)](arg0_1, arg1_1, buf2, 
            32768, 256, XBLOCK=32, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((32768,), (1,), torch.float32)
        triton_per_fused_abs_sub_sum_2[grid(32768)](arg0_1, arg1_1, buf3, 
            32768, 256, XBLOCK=32, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4
        del buf4
        triton_per_fused_abs_sub_sum_3[grid(1)](buf5, buf2, buf3, 1, 256,
            XBLOCK=1, num_warps=2, num_stages=1)
        del buf2
        del buf3
    return buf5,


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
