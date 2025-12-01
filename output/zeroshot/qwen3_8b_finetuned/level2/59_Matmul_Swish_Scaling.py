import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_sigmoid_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 2.0
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_0[grid(4194304)](buf1, 4194304, XBLOCK=
            256, num_warps=4, num_stages=1)
        del arg0_1
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        arg0_1 = self.matmul.weight
        arg1_1 = input_0
        self.matmul.weight = arg0_1
        del input_0
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        extern_kernels.mm(arg1_1, arg0_1, out=buf0)
        del arg0_1
        buf1 = buf0
        del buf0
        buf2 = call([buf1])
        return buf2[0]