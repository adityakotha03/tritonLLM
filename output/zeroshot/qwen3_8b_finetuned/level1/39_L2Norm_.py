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
def triton_per_fused_div_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 // 65535), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.where(xmask, tmp3, 0)
    tmp5 = tl.sum(tmp4, 0)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([XBLOCK, 1], 0, tl.int32)
    tmp8 = triton_helpers.promote_to_tensor(tl.where(xmask, tmp7, tmp6))
    tmp9 = tl.load(in_ptr0 + x0, xmask)
    tmp10 = tmp9 / tmp8
    tl.store(out_ptr0 + x0, tmp10, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 65535), (65535, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 65535), (65535, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_0[grid(32768)](arg0_1, buf0,
            32768, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(ModelNew, self).__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]