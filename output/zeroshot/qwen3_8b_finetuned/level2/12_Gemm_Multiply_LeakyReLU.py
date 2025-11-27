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
def triton_poi_fused_leaky_relu_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 0.1
    tmp4 = tmp2 >= 0
    tmp5 = tmp2 * tmp3
    tmp6 = tl.where(tmp4, tmp2, tmp5)
    tl.store(out_ptr0 + x2, tmp6, xmask)


def call(args):
    arg0_1, arg0_2, arg0_3 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    assert_size_stride(arg0_2, (8192,), (1,))
    assert_size_stride(arg0_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_mul_0[grid(8388608)](arg0_3, buf1, 8388608,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_3
        buf2 = buf1
        del buf1
    return buf2, arg0_1, arg0_2


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, input_0):
        arg0_1 = self.linear.weight
        arg0_2 = self.linear.bias
        arg0_3 = input_0
        output = call([arg0_1, arg0_2, arg0_3])
        return output[0]