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
def triton_poi_fused_add_arange_mul_rsub_sub_0(in_out_ptr0, in_ptr0, in_ptr1,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, rnumel)[None, :]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    r2 = rindex
    x0 = xindex
    y0 = rindex // 16
    y1 = rindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x0 + 4096 * y1 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + r2, None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + r2, None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = y0 <= x0
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (r2 + 256 * x0), tmp5, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((16, 256, 16), (4096, 1, 256), torch.float32
            )
        get_raw_stream(0)
        triton_poi_fused_add_arange_mul_rsub_sub_0[grid(16)](buf1, buf1,
            buf1, buf2, 16, 256, XBLOCK=16, num_warps=4, num_stages=1)
        buf3 = buf1
        del buf1
    return buf3,


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