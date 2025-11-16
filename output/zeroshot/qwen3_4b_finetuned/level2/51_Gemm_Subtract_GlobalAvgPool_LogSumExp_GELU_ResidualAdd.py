import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
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
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_gelu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 - tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp2 * tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp4
    tmp7 = tmp6 * tmp2
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused_global_mean_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (4096 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8192 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (12288 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_logsumexp_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tmp0 - tmp2
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = triton_helpers.promote_to_tensor(tl.broadcast_to(tmp3, [XBLOCK]))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.where(xmask, tmp6, float('-inf'))
    tmp9 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp10 = tmp2 + tmp9
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
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(2048)](buf0, primals_2, buf1, 2048,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        triton_poi_fused_global_mean_2[grid(1024)](buf1, buf2, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = reinterpret_tensor(buf2, (2048, 1), (1, 1), 0)
        del buf2
        triton_poi_fused_logsumexp_3[grid(2048)](buf1, buf3, 2048, XBLOCK=
            256, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((2048, 1), (1, 1), torch.float32)
        triton_poi_fused_gelu_1[grid(2048)](buf3, buf4, 2048, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf3
    return buf4, primals_3, primals_4, buf1


class ModelNew(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        primals_4 = self.subtract
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
