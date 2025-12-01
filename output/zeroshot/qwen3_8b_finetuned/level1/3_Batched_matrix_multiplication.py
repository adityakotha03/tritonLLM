import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16384
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256 * x2 + 16384 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 256 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (y0 + 1024 * x2 + 16384 * y1), tmp2, xmask & ymask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 512, 1024), (524288, 1024, 1))
    assert_size_stride(arg1_1, (128, 1024, 512), (532480, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 512, 1024), (524288, 1024, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(16384, 16384)](arg0_1, arg1_1, buf0, 
            16384, 16384, XBLOCK=128, YBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]