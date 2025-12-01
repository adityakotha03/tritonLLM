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
def triton_poi_fused_abs_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2147483680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.abs(tmp0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 65535.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 65535), (65535, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 65535), (65535, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_abs_div_0[grid(2147483680)](arg0_1, buf0, 2147483680,
            XBLOCK=512, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


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