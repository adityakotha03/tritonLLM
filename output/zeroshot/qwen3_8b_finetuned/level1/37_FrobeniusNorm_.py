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
def triton_per_fused_div_norm_fro_0(in_out_ptr0, in_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1887436480
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + r1, None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = libdevice.sqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_per_fused_div_norm_fro_1(in_out_ptr0, in_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1887436480
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp2 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp2 / tmp1
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (18874368, 294912, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_div_norm_fro_0[grid(1887436480)](buf1, arg0_1, 
            1887436480, 1, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        buf2 = empty_strided_cuda((112, 64, 512, 512), (18874368, 294912,
            512, 1), torch.float32)
        triton_per_fused_div_norm_fro_1[grid(1887436480)](buf2, buf1, 
            1887436480, 1, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf1
    return buf2,


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