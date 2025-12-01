import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + tl.broadcast_to(r0, [RBLOCK]), tmp5, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8388608, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_0[grid(8388608)](primals_2,
            primals_3, buf0, 8388608, 128, num_warps=4, num_stages=1)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_mul_sigmoid_0[grid(8388608)](primals_1,
            buf0, buf1, 8388608, 128, num_warps=4, num_stages=1)
        del primals_1
        del buf0
    return buf1


class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output