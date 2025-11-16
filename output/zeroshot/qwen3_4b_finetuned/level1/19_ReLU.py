import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 >= tmp1
    tmp3 = 0.0
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tl.store(in_out_ptr0 + tl.broadcast_to(r0, [RBLOCK]), tmp6, None)
    tl.store(out_ptr0 + tl.broadcast_to(r0, [RBLOCK]), tmp2, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = arg0_1
        buf1 = empty_strided_cuda((4096, 393216), (393216, 1), torch.bool)
        get_raw_stream(0)
        triton_per_fused_relu_threshold_backward_0[grid(256)](buf0, arg0_1,
            buf1, 256, 256, num_warps=2, num_stages=1)
        del arg0_1
    return buf0, buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
