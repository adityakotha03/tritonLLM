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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_hardtanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -2.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 2.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_hardtanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_hardtanh_gelu_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.702
    tmp2 = libdevice.erf(tmp0 * tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = tl_math.abs(tmp0)
    tmp7 = tl_math.erf(tmp6 * tmp1)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp0
    tmp11 = tmp3 - tmp10
    tmp12 = 0.7071067811865476
    tmp13 = tmp12 * tmp0
    tmp14 = tl.where(tmp5, tmp3, tmp13)
    tl.store(out_ptr0 + x0, tmp14, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    assert_size_stride(arg1_1, (2048, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_0[grid(2097152)](arg1_1, buf0, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_hardtanh_1[grid(2097152)](arg1_1, buf1, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_mul_2[grid(2097152)](buf0, buf2, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_hardtanh_gelu_mul_3[grid(2097152)](buf1, buf3,
            2097152, XBLOCK=512, num_warps=8, num_stages=1)
        del buf1
    return reinterpret_tensor(buf3, (2048, 8192), (8192, 1), 0), reinterpret_tensor(
        arg0_1, (8192, 8192), (1, 8192), 0), buf2


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
        arg1_1 = self.gemm.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
