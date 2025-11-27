import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_mean_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4096
    RBLOCK: tl.constexpr = 4095
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4096 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_per_fused_mean_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl
    .constexpr):
    RBLOCK: tl.constexpr = 4095
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 4095.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp5, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (16777216, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_mean_0[grid(4096)](arg0_1, buf0, 4096, 4095,
            XBLOCK=8, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1
        del buf1
        triton_per_fused_mean_1[grid(1)](buf2, buf0, 1, 4095, XBLOCK=1,
            num_warps=2, num_stages=1)
        del buf0
    return buf2,


class ModelNew(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
