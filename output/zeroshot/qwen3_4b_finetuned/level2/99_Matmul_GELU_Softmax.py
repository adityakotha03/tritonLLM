import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 83980864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tmp2 * tmp2
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp4, tmp2, tmp10)
    tl.store(in_out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused_softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 83980864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 8192 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 8192 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 8192 * x1), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr0 + (4 + 8192 * x1), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr0 + (5 + 8192 * x1), xmask, eviction_policy='evict_l'
        ast')
    tmp12 = tl.load(in_ptr0 + (6 + 8192 * x1), xmask, eviction_policy='evict_l'
        ast')
    tmp14 = tl.load(in_ptr0 + (7 + 8192 * x1), xmask, eviction_policy='evict_l'
        ast')
    tmp1 = tmp1 + tmp2
    tmp3 = tmp4 + tmp1
    tmp5 = tmp6 + tmp3
    tmp7 = tmp8 + tmp5
    tmp9 = tmp10 + tmp7
    tmp11 = tmp12 + tmp9
    tmp13 = tmp14 + tmp11
    tmp15 = tmp0 - tmp13
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp1 * tmp16
    tmp18 = tmp3 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tmp5 * tmp16
    tmp21 = tmp19 + tmp20
    tmp22 = tmp7 * tmp16
    tmp23 = tmp21 + tmp22
    tmp24 = tmp9 * tmp16
    tmp25 = tmp23 + tmp24
    tmp26 = tmp11 * tmp16
    tmp27 = tmp25 + tmp26
    tmp28 = tmp13 * tmp16
    tmp29 = tmp27 + tmp28
    tmp30 = 0.5 / tmp29
    tl.store(out_ptr0 + x2, tmp30, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(83980864)](buf0, primals_1, 83980864,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_softmax_1[grid(83980864)](buf0, buf1, 83980864,
            XBLOCK=256, num_warps=4, num_stages=1)
    return buf1, primals_2, primals_3, buf0


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
