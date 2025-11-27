import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_abs_mean_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 65535
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 65535 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_per_fused_abs_mean_1(in_out_ptr0, in_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 65535
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(0, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp7 = tl.where(0, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tmp4 / tmp8
    tmp10 = tl.full([1, 1], 0, tl.int32)
    tmp11 = tmp10 < tmp9
    tmp12 = tl.load(in_ptr0 + tl.broadcast_to(r0, [XBLOCK, RBLOCK]), tmp11,
        other=0.0)
    tmp13 = tl.where(0, tmp12, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp14 / tmp8
    tmp16 = tl.where(0, tmp1, 0)
    tmp17 = tmp16 / tmp8
    tmp18 = tl.where(0, tmp17, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp15 / tmp19
    tmp21 = tl.where(0, tmp0, 0)
    tmp22 = tmp21 / tmp19
    tmp23 = tl.where(0, tmp22, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp20 / tmp24
    tmp26 = tmp9 - tmp25
    tmp27 = tl.full([1, 1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp28, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 65535), (65535, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_per_fused_abs_mean_0[grid(32768)](arg0_1, buf0, 32768, 65535,
            XBLOCK=64, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1
        del buf1
        triton_per_fused_abs_mean_1[grid(1)](buf2, buf0, 1, 65535, XBLOCK=1,
            num_warps=2, num_stages=1)
        del buf0
    return buf2,


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
