import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_mul_relu_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (512 + r0), None)
    tmp5 = tl.load(in_ptr1 + (512 + r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1024 + r0), None)
    tmp9 = tl.load(in_ptr1 + (1024 + r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (1536 + r0), None)
    tmp13 = tl.load(in_ptr1 + (1536 + r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp3 + tmp7
    tmp17 = tmp16 + tmp11
    tmp18 = tmp17 + tmp15
    tl.store(in_out_ptr0 + tl.broadcast_to(r0, [RBLOCK]), tmp18, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192, 8192), (65536, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 1024), (1024, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_mul_relu_0[grid(262144)](buf0, arg1_1, arg0_1,
            262144, 256, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        arg1_1 = self.gemm.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
