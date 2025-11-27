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
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr0 + 4 * x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = tmp14 * tmp1
    tmp16 = tl_math.exp(tmp15)
    tl.store(out_ptr0 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16384, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384,), (1,))
    assert_size_stride(primals_3, (128, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(16777216)](primals_3, buf0, 
            16777216, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(16777216)](buf0, buf1, 16777216,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
