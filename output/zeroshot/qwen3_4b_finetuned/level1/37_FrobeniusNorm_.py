import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_div_linalg_vector_norm_0(in_out_ptr0, in_ptr0, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = tmp0 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp5, None)
    tl.store(in_ptr0 + tl.broadcast_to(r0, [XBLOCK, RBLOCK]), tmp6, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (16777216, 262144, 512,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_0[grid(1)](buf1, arg0_1, 1,
            64, XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
