import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4095
    x0 = xindex % 4095
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 128 * x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 128 * x0 + 4095), xmask)
    tmp3 = tl.load(in_ptr0 + (x1 + 128 * x0 + 8190), xmask)
    tmp5 = tl.load(in_ptr0 + (x1 + 128 * x0 + 12285), xmask)
    tmp7 = tl.load(in_ptr0 + (x1 + 128 * x0 + 16380), xmask)
    tmp9 = tl.load(in_ptr0 + (x1 + 128 * x0 + 20475), xmask)
    tmp11 = tl.load(in_ptr0 + (x1 + 128 * x0 + 24570), xmask)
    tmp13 = tl.load(in_ptr0 + (x1 + 128 * x0 + 28665), xmask)
    tmp15 = tl.load(in_ptr0 + (x1 + 128 * x0 + 32760), xmask)
    tmp17 = tl.load(in_ptr0 + (x1 + 128 * x0 + 36855), xmask)
    tmp23 = tmp0.to(tl.int32)
    tmp24 = tmp1.to(tl.int32)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = tmp3.to(tl.int32)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = tmp5.to(tl.int32)
    tmp29 = triton_helpers.maximum(tmp27, tmp28)
    tmp30 = tmp7.to(tl.int32)
    tmp31 = triton_helpers.maximum(tmp29, tmp30)
    tmp32 = tmp9.to(tl.int32)
    tmp33 = triton_helpers.maximum(tmp31, tmp32)
    tmp34 = tmp11.to(tl.int32)
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = tmp13.to(tl.int32)
    tmp37 = triton_helpers.maximum(tmp35, tmp36)
    tmp38 = tmp15.to(tl.int32)
    tmp39 = triton_helpers.maximum(tmp37, tmp38)
    tmp40 = tmp17.to(tl.int32)
    tmp41 = triton_helpers.maximum(tmp39, tmp40)
    tl.store(out_ptr0 + x2, tmp41, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (16794240, 4095, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4095, 1), (4095, 1, 4095), torch.int32
            )
        get_raw_stream(0)
        triton_poi_fused_max_0[grid(524288)](arg0_1, buf0, 524288, XBLOCK=
            256, num_warps=4, num_stages=1)
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
