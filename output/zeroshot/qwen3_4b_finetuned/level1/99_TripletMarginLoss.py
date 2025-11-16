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
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 2
    tmp0 = x3
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (8192 * x1 + x3 % 8192), tmp5, eviction_policy=
        'evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (8192 * x1 + x3 % 8192), tmp5, eviction_policy=
        'evict_last', other=0.0)
    tmp8 = tmp6 - tmp7
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tl.load(in_ptr2 + (8192 * x1 + x3 % 8192), tmp5, eviction_policy
        ='evict_last', other=0.0)
    tmp11 = tmp6 - tmp10
    tmp12 = tl_math.abs(tmp11)
    tmp13 = tl.where(tmp5, tmp9, tmp12)
    tl.store(out_ptr0 + x3, tmp13, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768, 1), (1, 1))
    assert_size_stride(arg2_1, (32768, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(262144)](arg0_1, arg1_1, arg2_1, buf0,
            262144, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
    return buf0,


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]
