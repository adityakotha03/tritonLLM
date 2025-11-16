import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_div_mul_rsub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + x2, xmask)
    tmp2 = tmp1 + tmp3
    tmp4 = tmp2 * 0.5
    tmp6 = tmp5 + tmp5
    tmp8 = tmp4 * tmp6
    tmp9 = tmp7 * tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp8 * tmp12
    tmp14 = tmp4 * tmp4
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = tmp0 * tmp15
    tmp17 = tmp8 * tmp16
    tmp18 = tmp17 * tmp12
    tmp19 = tmp13 - tmp18
    tl.store(out_ptr0 + x2, tmp19, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_mul_rsub_0[grid(16777216)](arg1_1,
            arg0_1, arg0_1, buf0, 16777216, XBLOCK=512, num_warps=8,
            num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
