import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4096
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 4095, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr0 + (x1 + 4096 * x0), tmp3 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp5 = tmp0 < 4096
    tmp6 = tl.full([1], 4095, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tl.load(in_ptr1 + (x1 + 4096 * x0), tmp5 & tmp7 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp9 = tl.where(tmp3, tmp8, 0.0)
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(16777216)](arg1_1, arg0_1, buf0,
            16777216, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return reinterpret_tensor(buf0, (4096, 4096), (4096, 1), 0),


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
