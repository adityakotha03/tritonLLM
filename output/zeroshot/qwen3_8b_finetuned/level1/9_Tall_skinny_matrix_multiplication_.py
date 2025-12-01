import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 32 * tl.load(in_ptr2 + x2, xmask,
        eviction_policy='evict_last')))
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp0
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32768, 16), (16, 1))
    assert_size_stride(primals_2, (16, 32768), (32768, 1))
    assert_size_stride(primals_3, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 32), (32, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(524288)](primals_1, primals_2,
            primals_3, buf0, 524288, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1, input_2):
        primals_1 = input_0
        primals_2 = input_1
        primals_3 = input_2
        output = call([primals_1, primals_2, primals_3])
        return output[0]