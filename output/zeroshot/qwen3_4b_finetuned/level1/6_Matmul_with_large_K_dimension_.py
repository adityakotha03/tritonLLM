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
def triton_poi_fused_native_batch_norm_backward_0(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp3 = tl.load(in_ptr1 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = tmp3 == tmp1
    tmp5 = tmp2 == tmp4
    tl.store(out_ptr0 + x0, tmp2, xmask)
    tl.store(out_ptr1 + x0, tmp5, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 524288), (524288, 1))
    assert_size_stride(arg1_1, (524288, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((256, 524288), (524288, 1), torch.bool)
        buf2 = empty_strided_cuda((256, 524288), (524288, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_native_batch_norm_backward_0[grid(131072)](buf0,
            buf1, buf2, buf1, 131072, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
    return buf2, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
