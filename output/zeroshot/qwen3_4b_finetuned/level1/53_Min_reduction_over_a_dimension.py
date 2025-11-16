import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_min_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 10240 % 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 4096000), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (10240 + x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (10240 + 4096 + x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (10240 + 4096 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp5 = triton_helpers.minimum(tmp1, tmp2)
    tmp6 = triton_helpers.minimum(tmp3, tmp4)
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tl_math.min(tmp0, tmp7)
    tl.store(out_ptr0 + x3, tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (1662560, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4096, 1), (4096, 1, 1662560), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused_min_0[grid(5242880)](arg0_1, buf0, 5242880,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
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
