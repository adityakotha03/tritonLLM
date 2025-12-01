import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (2048 + x0 + 4096 * x2), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 4096 * x2), xmask, eviction_policy
        ='evict_last')
    tmp5 = tl.load(in_ptr0 + (6144 + x0 + 4096 * x2), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 > tmp0
    tmp4 = tl.where(tmp2, tmp1, tmp0)
    tmp6 = tmp3 > tmp4
    tmp7 = tl.where(tmp6, tmp3, tmp4)
    tmp8 = tmp5 > tmp7
    tmp9 = tl.where(tmp8, tmp5, tmp7)
    tl.store(out_ptr0 + x2, tmp9, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (1638400, 4095, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_0[grid(128)](arg0_1, buf0, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]