import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__scaled_dot_product_attention_0(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 32
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask)
    tmp3 = tl.load(in_ptr2 + x3, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


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
        get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_attention_0[grid(524288)](arg0_1,
            arg1_1, arg2_1, buf0, 524288, XBLOCK=512, num_warps=8, num_stages=1
            )
        del arg0_1
        del arg1_1
        del arg2_1
    return buf0,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]
