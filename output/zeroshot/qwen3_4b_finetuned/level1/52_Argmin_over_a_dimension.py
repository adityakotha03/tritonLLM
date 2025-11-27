import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_argmin_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp1 == tmp1
    tmp3 = tmp0 < tmp1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp4 == tmp4
    tmp6 = tmp0 < tmp4
    tmp7 = tmp5 & tmp6
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp8 == tmp8
    tmp10 = tmp0 < tmp8
    tmp11 = tmp9 & tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = tmp12 == tmp12
    tmp14 = tmp0 < tmp12
    tmp15 = tmp13 & tmp14
    tmp16 = tl.full([1], 4, tl.int64)
    tmp17 = tmp16 == tmp16
    tmp18 = tmp0 < tmp16
    tmp19 = tmp17 & tmp18
    tmp20 = tmp15 | tmp19
    tmp21 = tmp11 | tmp20
    tmp22 = tmp7 | tmp21
    tmp23 = tmp3 | tmp22
    tmp24 = tl.where(tmp23, tmp0, tmp1)
    tl.store(out_ptr0 + x2, tmp24, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (16794240, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4096, 1), (4096, 1, 16794240), torch
            .int64)
        get_raw_stream(0)
        triton_poi_fused_argmin_0[grid(524288)](arg0_1, buf0, 524288,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
