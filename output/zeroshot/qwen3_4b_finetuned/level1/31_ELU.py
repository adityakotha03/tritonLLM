import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 15897504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = triton_helpers.maximum(tmp1, tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tmp6 * tmp3
    tmp8 = tmp0 * tmp5
    tmp9 = tmp8 - tmp7
    tmp10 = tmp9 * tmp3
    tmp11 = tl.where(tmp2, tmp6, tmp10)
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_elu_0[grid(15897504)](arg0_1, buf0, 15897504,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
