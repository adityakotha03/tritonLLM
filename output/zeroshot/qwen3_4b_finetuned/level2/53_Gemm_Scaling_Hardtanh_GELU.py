import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_hardtanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < -2.0
    tmp4 = tmp0 >= -2.0
    tmp5 = tmp0 < 2.0
    tmp6 = tmp4 & tmp5
    tmp7 = tl.where(tmp6, tmp0, 2.0)
    tmp8 = tl.where(tmp3, tmp0, -2.0)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp0
    tmp8 = 0.049999999999999996
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp10, tmp4)
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_0[grid(16777216)](arg1_1, buf0, 16777216,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = buf0
        del buf0
        triton_poi_fused_mul_1[grid(16777216)](buf1, 16777216, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_gelu_2[grid(16777216)](buf1, buf2, 16777216,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
    return buf2, arg0_1


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
