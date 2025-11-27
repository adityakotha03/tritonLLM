import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_selu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1592524160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 1.0507647162122208, tl.int32)
    tmp2 = tl.full([1], 1.673267430110121, tl.int32)
    tmp3 = tl_math.exp(tmp0)
    tmp4 = tmp3 - 1.0
    tmp5 = tmp2 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = tl.full([1], 0.0, tl.int32)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.where(tmp8, tmp6, tmp0)
    tl.store(out_ptr0 + x0, tmp9, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_selu_0[grid(1592524160)](arg0_1, buf0, 1592524160,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]