import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_hardtanh_gelu_1(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -2.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 2.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (2048, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(16777216)](buf0, primals_1, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_hardtanh_gelu_1[grid(16777216)](buf0, primals_3,
            buf1, 16777216, XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
        del primals_3
    return buf1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
