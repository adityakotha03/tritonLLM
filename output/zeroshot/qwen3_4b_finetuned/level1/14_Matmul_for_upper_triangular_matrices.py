import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_triu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4096
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = x1
    tmp1 = x0
    tmp2 = tmp0 >= tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_triu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4096
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = x1
    tmp1 = x0
    tmp2 = tmp0 >= tmp1
    tmp3 = 0.0
    tmp4 = tl.where(tmp2, tmp3, tmp1)
    tmp5 = tmp4 >= tmp3
    tl.store(out_ptr0 + x2, tmp5, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_triu_0[grid(16777216)](arg0_1, buf0, 16777216,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf1 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bool)
        triton_poi_fused_triu_0[grid(16777216)](arg1_1, buf1, 16777216,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg1_1, out=buf2)
        del arg0_1
        buf3 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bool)
        triton_poi_fused_triu_1[grid(16777216)](buf2, buf3, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf2
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
