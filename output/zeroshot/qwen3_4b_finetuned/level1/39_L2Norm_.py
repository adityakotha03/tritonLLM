import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_div_linalg_vector_norm_0(in_ptr0, out_ptr1, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 65536
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
    tl.store(out_ptr1 + tl.broadcast_to(r0, [XBLOCK, RBLOCK]), tmp6, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 65535), (65535, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((32768, 65535), (65535, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_0[grid(1)](arg0_1, buf1, 1,
            65536, XBLOCK=1, num_warps=8, num_stages=1)
        del arg0_1
    return buf1,


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
