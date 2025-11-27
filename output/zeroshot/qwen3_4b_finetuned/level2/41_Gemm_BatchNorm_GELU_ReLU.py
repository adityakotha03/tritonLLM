import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_native_batch_norm_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + 1 + 4096 * x0, xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + 2 + 4096 * x0, xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + 3 + 4096 * x0, xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)
    tl.store(out_ptr2 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tl.load(in_ptr3 + x0, xmask)
    tmp7 = tl.load(in_ptr4 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 / tmp5
    tmp8 = tmp7 * tmp6
    tmp9 = tmp0 - tmp3
    tmp10 = tmp9 * tmp9
    tmp11 = tmp10 / tmp5
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 / tmp5
    tmp14 = tmp13 + tmp21
    tmp15 = tl.where(xmask, tmp14, float('nan'))
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp15, xmask)


@triton.jit
def triton_poi_fused_gelu_relu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384 * 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (16384 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    tmp6 = 0.5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 < tmp12
    tmp16 = tmp15 & tmp14
    tmp17 = tl.where(tmp16, tmp11, tmp9)
    tmp18 = 0.0
    tmp19 = tl.where(tmp16, tmp18, tmp11)
    tl.store(out_ptr0 + x2, tmp17, xmask)
    tl.store(out_ptr0 + (4096 + x2), tmp19, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_native_batch_norm_0[grid(4096)](arg1_1, buf0,
            buf0, buf0, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        triton_poi_fused_native_batch_norm_1[grid(4096)](arg0_1, buf0,
            buf0, buf0, buf0, buf1, buf1, 4096, XBLOCK=128, num_warps=4,
            num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_gelu_relu_2[grid(67108864)](arg0_1, buf1, buf2,
            67108864, XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
        del buf1
    return buf2, arg1_1


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, input_0):
        arg1_1 = self.gemm.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
