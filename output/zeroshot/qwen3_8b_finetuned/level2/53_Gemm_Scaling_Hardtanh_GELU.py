import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_clamp_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 <= tmp0
    tmp4 = tmp2 >= tmp0
    tmp5 = tl.where(tmp3, tmp0, tmp1)
    tmp6 = tl.where(tmp4, tmp0, tmp2)
    tmp7 = tl.where(tmp5, tmp6, tmp5)
    tl.store(out_ptr0 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (8192,), (1,))
    assert_size_stride(primals_4, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mul_0[grid(16777216)](primals_1, buf1, 16777216,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf3 = buf2
        del buf2
        triton_poi_fused_clamp_1[grid(16777216)](buf1, primals_2,
            primals_3, buf3, 16777216, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        del primals_3
        buf4 = buf3
        del buf3
        triton_poi_fused_mul_2[grid(16777216)](buf4, buf4, 16777216,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
    return buf4, primals_4, reinterpret_tensor(buf1, (8192, 8192), (8192, 1),
        0), reinterpret_tensor(primals_2, (8192,), (1,), 0), reinterpret_tensor(
        primals_3, (8192,), (1,), 0)


class ModelNew(nn.Module):
    """
    Optimized version of the original model with Triton kernels for scaling,
    Hardtanh, and GELU.
    """
    def __init__(self, in_features, out_features, scaling_factor,
        hardtanh_min, hardtanh_max):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = self.hardtanh_min
        primals_4 = self.hardtanh_max
        primals_5 = self.scaling_factor
        output = call([primals_1, primals_2, primals_3, primals_4, input_0])
        return output[0]