import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused__logsumexp_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp7 = tl_math.log(tmp6)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp7, None)
    tl.store(out_ptr0 + tl.broadcast_to(r0, [XBLOCK, RBLOCK]), tmp2, None)


@triton.jit
def triton_poi_fused_add_gelu_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + x0, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (2048, 8192), (8192, 1))
    assert_size_stride(primals_4, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (8192, 8192), (1, 8192), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__logsumexp_0[grid(1)](buf2, buf0, primals_4, buf3,
            1, 8192, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_gelu_1[grid(16384)](buf2, primals_3, buf4, 
            16384, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
    return buf4, primals_3, buf0, buf3


class ModelNew(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, input_0):
        primals_4 = self.subtract
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
