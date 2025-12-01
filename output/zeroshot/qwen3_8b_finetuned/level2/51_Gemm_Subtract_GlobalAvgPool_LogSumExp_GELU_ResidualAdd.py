import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (2048, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192, 8192), (8192, 1))
    assert_size_stride(primals_3, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_3, buf0, reinterpret_tensor(primals_1,
            (8192, 8192), (1, 8192), 0), alpha=1, beta=1, out=buf1)
        del primals_3
        buf2 = empty_strided_cuda((2048, 1), (1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__log_sum_exp_1[grid(2048)](buf1, buf2, 2048, XBLOCK
            =128, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((2048, 1), (1, 1), torch.float32)
        extern_kernels.gelu(buf2, out=buf3)
        buf4 = buf2
        del buf2
        extern_kernels.addmm(primals_3, buf3, reinterpret_tensor(primals_1,
            (1, 8192), (1, 1), 0), alpha=1, beta=1, out=buf4)
        del primals_3
        buf5 = empty_strided_cuda((2048, 1), (1, 1), torch.float32)
        triton_poi_fused_add_0[grid(2048)](buf4, primals_1, buf5, 2048,
            XBLOCK=128, num_warps=4, num_stages=1)
    return buf5, primals_1, buf0, buf1, buf3, buf4


class ModelNew(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, input_0):
        primals_2 = self.gemm.weight
        primals_3 = self.gemm.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]