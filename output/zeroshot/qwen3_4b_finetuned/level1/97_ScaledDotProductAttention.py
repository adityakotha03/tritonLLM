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
def triton_poi_fused__scaled_dot_product_attention_0(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp10 = tl.load(in_ptr2 + x2, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tmp4 * tmp1
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.where(xmask, tmp6, float('inf'))
    tmp9 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp11 = tmp5 - tmp9
    tmp12 = tmp10 * tmp1
    tmp13 = tmp12 - tmp3
    tmp14 = tmp13 * tmp1
    tmp15 = tmp14 - tmp9
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp16 / tmp20
    tl.store(out_ptr0 + x2, tmp9, xmask)
    tl.store(out_ptr1 + x2, tmp2, xmask)
    tl.store(out_ptr2 + x2, tmp21, xmask)


@triton.jit
def triton_poi_fused__scaled_dot_product_attention_1(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 / tmp7
    tmp8 = tmp0 * tmp6
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 32, 512, 1024), (16777216, 524288, 1024,
        1))
    assert_size_stride(arg1_1, (32, 32, 512, 1024), (16777216, 524288, 1024,
        1))
    assert_size_stride(arg2_1, (32, 32, 512, 1024), (16777216, 524288, 1024,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288,
            1024, 1), torch.float16)
        buf1 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288,
            1024, 1), torch.float16)
        buf2 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288,
            1024, 1), torch.float16)
        get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_attention_0[grid(524288)](arg0_1,
            arg1_1, arg2_1, buf0, buf1, buf2, 524288, XBLOCK=512, num_warps=
            8, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
        buf3 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288,
            1024, 1), torch.float16)
        triton_poi_fused__scaled_dot_product_attention_1[grid(524288)](buf2,
            buf1, buf0, buf1, buf3, buf3, 524288, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf1
        del buf2
    return buf3,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]
