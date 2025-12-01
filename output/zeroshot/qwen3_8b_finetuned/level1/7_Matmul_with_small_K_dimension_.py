import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    x2 = xindex
    y3 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (r0 + 64 * x2), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0 + 64 * x2 + 131072 * y3), xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x2, tmp6, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (131072, 64), (64, 1))
    assert_size_stride(arg1_1, (64, 131072), (1, 64))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((131072, 131072), (131072, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_add_0[grid(131072)](buf1, arg0_1, arg1_1, 131072,
            64, XBLOCK=32, num_warps=8, num_stages=1)
        del arg0_1
        del arg1_1
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]