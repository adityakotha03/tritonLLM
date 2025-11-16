import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_matmul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 1024
    x0 = xindex // 16384
    x4 = xindex % 768
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5 + tmp7
    tmp8 = tmp0 * tmp6
    tl.store(out_ptr0 + (x0 + 16384 * x4 + 16777216 * x3), tmp8, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(arg1_1, (2048, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1024, 768), (786432, 768, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_matmul_0[grid(16777216)](arg1_1, arg0_1, buf0, 
            16777216, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
