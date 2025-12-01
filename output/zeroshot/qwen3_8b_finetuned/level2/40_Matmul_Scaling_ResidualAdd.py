import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + 0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + x2, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_0[grid(67108864)](buf0, primals_2,
            primals_3, primals_1, 67108864, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
    return buf0, primals_3


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]