import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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
    x1 = xindex // 4096 % 2048
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 8192 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp5 = tl.load(in_ptr0 + (6144 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp4 = tmp1 + tmp2
    tmp6 = tmp3 + tmp5
    tmp7 = tmp4 + tmp6
    tmp8 = tmp0 - tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096 % 2048
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 8192 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp4 = tl.load(in_ptr0 + (6144 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp5 = tmp1 + tmp2
    tmp6 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096 % 2048
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 8192 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp4 = tl.load(in_ptr0 + (6144 + x1 + 8192 * x2), xmask, eviction_policy
        ='evict_last')
    tmp5 = tmp1 + tmp2
    tmp6 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tmp9 = tmp8 * tmp8
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4096 % 2048
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp3 = tl.load(in_ptr2 + x2, xmask)
    tmp5 = tl.load(in_ptr3 + x2, xmask)
    tmp7 = tl.load(in_ptr4 + x2, xmask)
    tmp9 = tl.load(in_ptr5 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp0 - tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = tmp12 == tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp14
    tl.store(out_ptr0 + (x0 + 4096 * x2), tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 2048), (2048, 1))
    assert_size_stride(arg1_1, (2048, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(16777216)](arg1_1, buf0, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(16777216)](arg1_1, buf1, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        triton_poi_fused__softmax_2[grid(16777216)](buf1, buf2, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        triton_poi_fused__softmax_3[grid(16777216)](arg0_1, buf0, buf1,
            buf2, buf2, buf2, buf3, 16777216, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf0
        del buf1
        del buf2
    return buf3, arg0_1


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
