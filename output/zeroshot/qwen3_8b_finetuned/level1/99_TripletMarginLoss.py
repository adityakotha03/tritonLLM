import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_triplet_loss_mean_0(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + r0 * 32768), None)
    tmp1 = tl.full([1], 32768, tl.int64)
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (r0), tmp2, None)


@triton.jit
def triton_poi_fused_triplet_loss_per_element_1(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    x3 = xindex
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + x3), None)
    tmp1 = tl.load(in_ptr1 + (r0 + x3), None)
    tmp2 = tl.load(in_ptr2 + (r0 + x3), None)
    tmp3 = tmp0 - tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp0 - tmp2
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 - tmp6
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tl.store(out_ptr0 + x3, tmp11, None)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 8192), (8192, 1))
    assert_size_stride(arg1_1, (32768, 8192), (8192, 1))
    assert_size_stride(arg2_1, (32768, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        buf1 = empty_strided_cuda((32768, 8192), (8192, 1), torch.float32)
        del arg0_1
        del arg1_1
        del arg2_1
        get_raw_stream(0)
        triton_poi_fused_triplet_loss_per_element_1[grid(262144)](arg0_1,
            arg1_1, arg2_1, buf1, 262144, 1, XBLOCK=128, num_warps=4,
            num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
        buf2 = buf1
        del buf1
        triton_poi_fused_triplet_loss_mean_0[grid(1)](buf2, buf0, 32768, 1,
            XBLOCK=1, num_warps=1, num_stages=1)
        del buf2
    return buf0,


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.loss_fn = margin

    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]