import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 66038400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x3 = xindex
    x4 = xindex
    x5 = xindex
    x6 = xindex
    x7 = xindex
    x8 = xindex
    x9 = xindex
    x1 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + x4, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + x5, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + x6, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + x7, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + x8, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + x9, xmask, eviction_policy='evict_last')
    tmp10 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp11 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp20 = tmp18 * tmp19
    tmp21 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp22 = tmp20 * tmp21
    tmp23 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp24 = tmp22 * tmp23
    tmp25 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp26 = tmp24 * tmp25
    tmp27 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp28 = tmp26 * tmp27
    tmp29 = 0.044715
    tmp30 = tmp28 * tmp29
    tmp31 = 2.0
    tmp32 = tmp30 + tmp31
    tmp33 = math.sqrt(2.0 / math.pi)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp36 = tmp28 + tmp35
    tmp37 = tl.tanh(tmp36)
    tmp38 = 1.0
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39 * tmp28
    tmp41 = 0.5
    tmp42 = tmp40 * tmp41
    tl.store(out_ptr0 + x0, tmp42, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(66038400)](arg0_1, buf0, 66038400,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]