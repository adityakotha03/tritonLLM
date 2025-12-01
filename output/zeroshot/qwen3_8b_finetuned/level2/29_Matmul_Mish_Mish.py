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
def triton_poi_fused_add_mish_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp2 < tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp5, 1.0)
    tmp7 = tmp0 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp0 * tmp10
    tmp12 = tmp11 - tmp5
    tmp13 = 2.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.tanh(tmp14)
    tmp16 = tmp15 * tmp11
    tl.store(out_ptr0 + x0, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mish_0[grid(8388608)](primals_3, buf0, 8388608,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_mish_0[grid(8388608)](buf0, buf1, 8388608,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]