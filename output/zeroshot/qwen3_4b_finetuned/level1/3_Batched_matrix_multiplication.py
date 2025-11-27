import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_bmm_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 2048 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 2048, 1024), (2097152, 1024, 1))
    assert_size_stride(arg1_1, (128, 1024, 2048), (2097152, 2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 2048, 1024), (2097152, 1024, 1),
            torch.float32)
        extern_kernels.bmm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((128, 2048, 1024), (2097152, 1024, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_bmm_0[grid(2097152)](buf0, buf0, buf1, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
    return buf1,


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
