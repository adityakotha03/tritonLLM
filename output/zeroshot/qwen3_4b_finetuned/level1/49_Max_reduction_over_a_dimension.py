import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 4096 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 4096 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (3 + x0 + 4096 * x1), xmask)
    tmp2 = tmp0 > tmp1
    tmp4 = tmp0 > tmp3
    tmp6 = tmp0 > tmp5
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.where(tmp8, tmp0, tmp1)
    tmp10 = tl.where(tmp8, tmp0, tmp3)
    tmp11 = tl.where(tmp8, tmp0, tmp5)
    tmp12 = tl.where(tmp8, tmp9, tmp10)
    tmp13 = tl.where(tmp8, tmp9, tmp11)
    tl.store(out_ptr0 + x2, tmp13, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (16793664, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4096, 4095), (16793664, 4096, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_0[grid(5242880)](arg0_1, buf0, 5242880,
            XBLOCK=1024, num_warps=4, num_stages=1)
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
