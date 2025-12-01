import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1625754240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tl.load(in_ptr1 + 0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp1 = tmp0 * tmp3
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 393216), (393216, 1))
    assert_size_stride(primals_2, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_0[grid(1625754240)](primals_1, primals_2,
            buf0, 1625754240, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, input_0):
        primals_2 = self.negative_slope
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]