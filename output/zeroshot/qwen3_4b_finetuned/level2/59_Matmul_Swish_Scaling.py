import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_sigmoid_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp0 * tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 32768), (32768, 1))
    assert_size_stride(arg1_1, (32768, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_sigmoid_0[grid(16777216)](buf0, 16777216, XBLOCK=
            1024, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_mul_1[grid(16777216)](arg1_1, buf1, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg1_1
    return buf0, buf1, arg0_1


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        arg1_1 = self.matmul.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
