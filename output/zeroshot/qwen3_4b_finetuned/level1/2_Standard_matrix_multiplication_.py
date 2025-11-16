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
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2048 % 4096
    x0 = xindex % 2048
    x2 = xindex // 8388608
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 4096, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 2048 * x1), tmp4 & xmask, other=0.0)
    tmp6 = x0
    tmp7 = tmp6 < 2048
    tmp8 = tl.load(in_ptr1 + (x0 + 2048 * x1 + 8388608 * x2), tmp7 & xmask,
        other=0.0)
    tmp9 = tl.load(in_ptr0 + (x0 + 2048 * (-4096 + x1) + 8388608 * x2),
        tmp4 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr1 + (x0 + 2048 * (-4096 + x1) + 8388608 * x2),
        tmp7 & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp5 * tmp8
    tmp13 = tmp11 * tmp5
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + x3, tmp14, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(8388608)](arg0_1, arg1_1, buf0, 8388608,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


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
