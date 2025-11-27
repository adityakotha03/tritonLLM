import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__triu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 <= tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__triu_add_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 4096 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 4096 * x1), xmask)
    tmp4 = tl.load(in_ptr1 + (4096 + x0 + 4096 * x1), xmask)
    tmp6 = tl.load(in_ptr0 + (8192 + x0 + 4096 * x1), xmask)
    tmp7 = tl.load(in_ptr1 + (8192 + x0 + 4096 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (12288 + x0 + 4096 * x1), xmask)
    tmp10 = tl.load(in_ptr1 + (12288 + x0 + 4096 * x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = tmp2 == tmp12
    tmp14 = tmp5 == tmp12
    tmp15 = tmp13 | tmp14
    tmp16 = tmp8 == tmp12
    tmp17 = tmp15 | tmp16
    tmp18 = tmp11 == tmp12
    tmp19 = tmp17 | tmp18
    tmp20 = 1.0
    tmp21 = tmp2 * tmp20
    tmp22 = tmp5 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp8 * tmp20
    tmp25 = tmp23 + tmp24
    tmp26 = tmp11 * tmp20
    tmp27 = tmp25 + tmp26
    tmp28 = tmp19 * tmp27
    tl.store(out_ptr0 + x2, tmp28, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused__triu_0[grid(16777216)](arg0_1, buf0, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bool)
        triton_poi_fused__triu_0[grid(16777216)](arg1_1, buf1, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        triton_poi_fused__triu_add_mul_1[grid(16777216)](arg0_1, arg1_1,
            buf2, 16777216, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf2, buf0, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
