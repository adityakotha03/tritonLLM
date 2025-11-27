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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr0 + 2 * x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 2 * x1), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp3 - tmp1
    tmp6 = tmp5 - tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 2 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 2 * x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_per_fused__log_softmax_add_div_mul_sub_2(in_out_ptr0, in_ptr0,
    in_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + 2 * r0, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 2 * r0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + 2 * r0, None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + 2 * r0), None, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.exp(tmp3)
    tmp4 = tmp1 + tmp2
    tmp5 = tl_math.log(tmp4)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl_math.exp(tmp9)
    tmp10 = tmp7 + tmp8
    tmp11 = tmp0 - tmp5
    tmp12 = tmp11 * tmp10
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 0.0001220703125
    tmp17 = tmp15 * tmp16
    tmp18 = 0.0
    tmp19 = tmp17 + tmp18
    tmp20 = 128.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([1], 0, tl.int32), tmp21, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16384,), (1,))
    assert_size_stride(arg1_1, (16384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(16384)](arg0_1, buf0, 16384,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((16384,), (1,), torch.float32)
        triton_poi_fused__softmax_0[grid(16384)](arg1_1, buf1, 16384,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((16384,), (1,), torch.float32)
        triton_poi_fused__softmax_1[grid(16384)](buf0, buf2, 16384, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((16384,), (1,), torch.float32)
        triton_poi_fused__softmax_1[grid(16384)](buf1, buf3, 16384, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf1
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4
        del buf4
        triton_per_fused__log_softmax_add_div_mul_sub_2[grid(1)](buf5, buf2,
            buf3, 1, 256, num_warps=2, num_stages=1)
        del buf2
        del buf3
    return buf5,


class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
